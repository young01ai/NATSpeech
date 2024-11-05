from data_gen.tts.txt_processors.base_text_processor import BaseTxtProcessor, register_txt_processors
from utils.text.text_encoder import PUNCS, is_sil_phoneme


class MixG2p:
    def __call__(self, text):
        prons = []
        for word in text.split(" "):
            prons.extend(word.split("-") + [" "])
        return prons[:-1]


@register_txt_processors("mix")
class TxtProcessor(BaseTxtProcessor):
    g2p = MixG2p()

    @staticmethod
    def preprocess_text(text):
        words = [word for word in text.strip().split(" ")]
        return " ".join(words[1:-2])  # NOTE: remove head and tail <sil>

    @classmethod
    def process(cls, txt, preprocess_args):
        txt = cls.preprocess_text(txt).strip()
        phs = cls.g2p(txt)
        txt_struct = [[w, []] for w in txt.split(" ")]
        i_word = 0
        for p in phs:
            if p == ' ':
                i_word += 1
            else:
                txt_struct[i_word][1].append(p)
        txt_struct = cls.postprocess(txt_struct, preprocess_args)
        return txt_struct, txt


if __name__ == '__main__':
    txt = "<sil> ni-i_l-i_h h-aa_l-o_l <br3> sh-ib_h-ib_l ji-eh_h-eh_l ! <br4> <sil>"
    txt = "<sil> zh-el_h-el_l <br0> h-el_l-nn_l <br2> N-AY1-S <br1> ga-a_h-a_l ! <br4> <sil>"
    txt_processor = TxtProcessor()
    txt_struct, txt = txt_processor.process(txt, {'with_phsep': False, 'add_eos_bos': True})
    print(txt_struct)
    print(txt)