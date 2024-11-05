import re
import unicodedata

from tn.chinese.normalizer import Normalizer as ZhNormalizer
zh_tn_model = ZhNormalizer()
from pypinyin import lazy_pinyin, Style
from jieba import posseg as psg
from data_gen.tts.txt_processors.tone_sandhi import ToneSandhi as ZhToneSandhi
tone_sandhi = ZhToneSandhi()

from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from utils.text.text_encoder import PUNCS, is_sil_phoneme


def _get_initials_finals(word):
    initials, finals = [], []
    origin_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    origin_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
    for c, v in zip(origin_initials, origin_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


class ZhG2p:
    def __call__(self, text):
        # preprocessing
        tokens = psg.lcut(text)
        tokens = tone_sandhi.pre_merge_for_modify(tokens)
        # steps
        prons = []
        for word, pos in tokens:
            if word in [" "] + list(PUNCS):
                prons.extend([word, " "])
            else:
                pron = []
                initials, finals = _get_initials_finals(word)
                finals = tone_sandhi.modified_tone(word, pos, finals)
                for ini, fin in zip(initials, finals):
                    pron.extend([ini, fin, " "])
                prons.extend(pron)

        return prons[:-1]


@register_txt_processors("zh")
class TxtProcessor(BaseTxtProcessor):
    g2p = ZhG2p()

    @staticmethod
    def preprocess_text(text):
        text = zh_tn_model.normalize(text).replace("\n", ".")
        text = re.sub(rf"[^ \u4e00-\u9fa5{PUNCS}]+", "", text)
        text = re.sub(f" ?([{PUNCS}]) ?", r"\1", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)
        # text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        # text = re.sub(rf"\s+", r" ", text)
        return text.replace(" ", "")

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in list(txt)]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt


if __name__ == '__main__':
    txt = "你好，世界！"
    txt_processor = TxtProcessor()
    txt_struct, txt = txt_processor.process(txt, {'with_phsep': False, 'add_eos_bos': False})
    print(txt_struct)
    print(txt)