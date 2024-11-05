from data_gen.tts.base_preprocess import BasePreprocessor


class LJPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/metadata.csv').readlines():
            item_name, _, txt = l.strip().split("|")
            wav_fn = f"{self.raw_data_dir}/wavs/{item_name}.wav"
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt}


class BiaobeiPreprocess(BasePreprocessor):
    def meta_data(self):
        with open(f'{self.raw_data_dir}/ProsodyLabeling/000001-010000.txt', encoding='utf-8') as f:
            bb_lines = f.readlines()[::2]
        for i, l in (enumerate([re.sub("\#\d+", "", l.split('\t')[1].strip()) for l in bb_lines])):
            item_name = f'{i+1:06d}'
            wav_fn = f'{self.raw_data_dir}/Wave/{i+1:06d}.wav'
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': l.strip()}


cmu_vowel = ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
cmu_vowel_with_stress = [f"{v}{s}" for v in cmu_vowel for s in range(3)]
cmu_consonant = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]
cmu_phoneme = cmu_consonant + cmu_vowel
cmu_phoneme_with_stress = cmu_consonant + cmu_vowel_with_stress

ice_vowel_zhcn = ["a", "aa", "eh", "el", "er", "i", "ib", "if", "ng", "nn", "o", "u", "v", "zzzar", "zzzer", "zzznr", "zzzor", "zzzur"]
ice_vowel_zhcn_with_tone = [f"{v}_{t}" for v in ice_vowel_zhcn for t in ["l", "m", "h"]]
ice_consonant_zhcn = ["b", "bi", "bu", "c", "ch", "chu", "cu", "d", "di", "du", "f", "fu", "g", "ga", "ge", "go", "gu", "h", "hu", "ji", "jv", "k", "ku", "l", "li", "lu", "lv", "m", "mi", "mu", "n", "ni", "nu", "nv", "p", "pi", "pu", "qi", "qv", "r", "ru", "s", "sh", "shu", "su", "t", "ti", "tu", "wu", "xi", "xv", "yi", "yv", "z", "zh", "zhu", "zu"]
ice_phoneme_zhcn = ice_vowel_zhcn + ice_consonant_zhcn
ice_phoneme_zhcn_with_tone = ice_consonant_zhcn + ice_vowel_zhcn_with_tone

# ice_vowel_mixlingual = ["a", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "el", "er", "ey", "i", "ib", "if", "ih", "iy", "ng", "nn", "o", "ow", "oy", "u", "uh", "uw", "v", "zzzar", "zzzer", "zzznr", "zzzor", "zzzur"]
# ice_consonant_mixlingual = ["b", "bi", "bu", "c", "ch", "chu", "cu", "d", "dd", "dh", "di", "du", "f", "ff", "fu", "g", "ga", "ge", "gg", "go", "gu", "h", "hh", "hu", "jh", "ji", "jv", "k", "kk", "ku", "l", "li", "lu", "lv", "m", "mi", "mu", "n", "ng", "ni", "nu", "nv", "p", "ph", "pi", "pu", "qi", "qv", "r", "rr", "ru", "s", "sh", "shu", "ssh", "su", "t", "th", "ti", "tsh", "tt", "tu", "v", "wu", "xi", "xv", "yi", "yv", "z", "zh", "zhu", "zu", "zz", "zzh"]
# ice_phoneme_mixlingual_with_tone = ['a_h', 'a_l', 'a_m', 'aa', 'aa_h', 'aa_l', 'aa_m', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'bi', 'bu', 'c', 'ch', 'chu', 'cu', 'd', 'dd', 'dh', 'di', 'du', 'eh', 'eh_h', 'eh_l', 'eh_m', 'el_h', 'el_l', 'el_m', 'er', 'er_h', 'er_l', 'er_m', 'ey', 'f', 'ff', 'fu', 'g', 'ga', 'ge', 'gg', 'go', 'gu', 'h', 'hh', 'hu', 'i_h', 'i_l', 'i_m', 'ib_h', 'ib_l', 'ib_m', 'if_h', 'if_l', 'if_m', 'ih', 'iy', 'jh', 'ji', 'jv', 'k', 'kk', 'ku', 'l', 'li', 'lu', 'lv', 'm', 'mi', 'mu', 'n', 'ng', 'ng_h', 'ng_l', 'ng_m', 'ni', 'nn_h', 'nn_l', 'nn_m', 'nu', 'nv', 'o_h', 'o_l', 'o_m', 'ow', 'oy', 'p', 'ph', 'pi', 'pu', 'qi', 'qv', 'r', 'rr', 'ru', 's', 'sh', 'shu', 'ssh', 'su', 't', 'th', 'ti', 'tsh', 'tt', 'tu', 'u_h', 'u_l', 'u_m', 'uh', 'uw', 'v', 'v_h', 'v_l', 'v_m', 'wu', 'xi', 'xv', 'yi', 'yv', 'z', 'zh', 'zhu', 'zu', 'zz', 'zzh', 'zzzar_h', 'zzzar_l', 'zzzar_m', 'zzzer_h', 'zzzer_l', 'zzzer_m', 'zzznr_h', 'zzznr_l', 'zzznr_m', 'zzzor_h', 'zzzor_l', 'zzzor_m', 'zzzur_h', 'zzzur_l', 'zzzur_m']

ice_break = ["br0", "br1", "br2", "br3", "br4"]
ice_silence = ["sil", "spn"] + ice_break
punctuation = [".", ",", ":", ";", "!", "?"]

def preprocess(sequence):
    phonemes = sequence.split(' ')
    assert phonemes[0] == 'sil' and phonemes[-1] == 'sil', f'Warning: sequence does not match the format'
    syllables, syllable = [], []
    for phn in phonemes:
        if phn in ['sil', 'spn'] or phn.startswith('br'):
            assert len(syllable) == 0, f'Error: sequence has wrong phonemes (1)'
            syllables.append(f'<{phn}>')
        elif phn in punctuation:
            assert len(syllable) == 0, f'Error: sequence has wrong phonemes (2)'
            syllables.append(phn)
        else:
            if phn in ice_phoneme_zhcn_with_tone:
                if len(syllable) == 0:
                    assert phn in ice_consonant_zhcn, f'Error: sequence has wrong phonemes (3) {phn}'
                elif len(syllable) <= 2:
                    assert phn in ice_vowel_zhcn_with_tone, f'Error: sequence has wrong phonemes (3) {phn}'
                else:
                    assert False, f'Error: sequence has wrong phonemes (4) {phn}'
                syllable.append(phn)
                if len(syllable) == 3:
                    syllables.append('-'.join(syllable))
                    syllable = []
            elif phn.split('_')[0] in cmu_phoneme_with_stress:
                if phn.split('_')[1] in ['S', 'B']:
                    assert len(syllable) == 0, f'Error: sequence has wrong phonemes (5) {phn}'
                elif phn.split('_')[1] in ['M', 'E']:
                    assert len(syllable) > 0, f'Error: sequence has wrong phonemes (5) {phn}'
                    # for i, p in enumerate(syllable):
                    #     if i == 0:
                    #         assert p.split('_')[1] == 'B', f'Error: sequence has wrong phonemes (5) {phn}'
                    #     else:
                    #         assert p.split('_')[1] == 'M', f'Error: sequence has wrong phonemes (5) {phn}'
                else:
                    assert False, f'Error: sequence has wrong phonemes (6) {phn}'
                syllable.append(phn.split('_')[0])
                if phn.split('_')[1] in ['S', 'E']:
                    syllables.append('-'.join(syllable))
                    syllable = []
            else:
                if phn in ['&', '-', '1', '2']:  # ignore some useless phonemes
                    continue
                assert False, f'Error: sequence has wrong phonemes (0) {phn}'
    return ' '.join(syllables)

class MixPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/metadata.csv').readlines():
            item_data = l.strip().split("|")
            assert len(item_data) == 3, print('please make sure for the input format')
            path, txt, phns = item_data
            wav_fn = f"{self.raw_data_dir}/wavs/{path}"
            item_name = '_'.join(path[:-4].split('/'))
            spk_name = path[:-4].split('/')[0]
            try:
                txt_preprocessed = preprocess(phns)
            except AssertionError as e:
                print(f"Processing {item_name} failed: {e}")
                continue
            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt_preprocessed, 'spk_name': spk_name}


if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    set_hparams(config=r'egs/datasets/audio/mix/base_text2mel.yaml', print_hparams=False)
    preprocessor = MixPreprocess()
    for item in preprocessor.meta_data():
        print(item)
        break