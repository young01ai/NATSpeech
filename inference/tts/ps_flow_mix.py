import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.portaspeech.portaspeech_flow import PortaSpeechFlow
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams

## Frontend Function
import os
import subprocess
from egs.datasets.audio.mix.preprocess import preprocess

FRONTEND_ROOT = '/mnt/data/digiman/zhouzhiyang/workspace/migrate/Frontend-v2-MixLingual'

def start_process():
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = os.path.join(FRONTEND_ROOT, 'libs')
    args = [os.path.join(FRONTEND_ROOT, 'frontend_main')]
    return subprocess.Popen(args,  # no args
                            stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=FRONTEND_ROOT,
                            shell=False, env=env, text=True)

def frontend_main(sequence):
    global process
    if process is None:
        process = start_process()
    sequence = sequence.strip().replace('\n', '')
    try:
        process.stdin.write(sequence + '\n')
        process.stdin.flush()  # flush the internal buffer
        sequence = process.stdout.readline()
        while '[' in sequence or ']' in sequence:
            sequence = process.stdout.readline()
    except Exception as e:
        print(f"Error during process execution: {e}")
        process = start_process()
        return None
    sequence = sequence.strip().replace('\n', '')
    return preprocess(sequence)

global process
process = start_process()
## Frontend Function

class PortaSpeechFlowInfer(BaseTTSInfer):
    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = PortaSpeechFlow(ph_dict_size, word_dict_size, self.hparams)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        with torch.no_grad():
            model.store_inverse_all()
        model.eval()
        return model

    def preprocess_input(self, inp):
        inp['text'] = frontend_main(inp['text'])
        return super().preprocess_input(inp)

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                infer=True,
                forward_post_glow=True,
                spk_id=sample.get('spk_ids')
            )
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    PortaSpeechFlowInfer.example_run()
