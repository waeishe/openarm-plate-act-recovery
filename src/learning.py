"""Actual LeRobot ACT training on aligned simulated expert demonstrations."""
import json,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType,PolicyFeature
from environment import ROOT,CAMERAS,sha
from openarm_vla.common import atomic_json

def make_config(chunk):
    return ACTConfig(input_features={'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(16,)),**{f'observation.images.{k}':PolicyFeature(type=FeatureType.VISUAL,shape=(3,128,160)) for k in CAMERAS}},output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(16,))},device='mps',push_to_hub=False,chunk_size=chunk,n_action_steps=10,dim_model=256,n_heads=8,dim_feedforward=1024,n_encoder_layers=2,n_decoder_layers=1,n_vae_encoder_layers=2,latent_dim=32,kl_weight=1.,dropout=.1,optimizer_lr=1e-4,optimizer_lr_backbone=1e-5)

class Data:
    def __init__(self,paths,stats=None):
        self.paths=paths;self.episodes=[]
        for p in paths:
            with np.load(p) as d:self.episodes.append({k:d[k] for k in d.files})
        if stats is None:
            state=np.concatenate([e['state'] for e in self.episodes]);action=np.concatenate([e['action'] for e in self.episodes])
            stats={k:v.tolist() for k,v in {'state_mean':state.mean(0),'state_std':np.maximum(state.std(0),.05),'action_mean':action.mean(0),'action_std':np.maximum(action.std(0),.05)}.items()}
        self.stats=stats;self.rng=np.random.default_rng(731)
    def batch(self,n,chunk,device):
        indices=self.rng.integers(0,len(self.episodes),n);states=[];actions=[];pads=[];images={k:[] for k in CAMERAS}
        for eidx in indices:
            e=self.episodes[eidx];t=int(self.rng.integers(0,len(e['state'])));ix=t+np.arange(chunk);pad=ix>=len(e['state']);ix=np.minimum(ix,len(e['state'])-1)
            states.append(e['state'][t]);actions.append(e['action'][ix]);pads.append(pad)
            for k in CAMERAS:images[k].append(e[k][t])
        s=self.stats
        batch={'observation.state':torch.tensor((np.stack(states)-s['state_mean'])/s['state_std'],dtype=torch.float32,device=device),'action':torch.tensor((np.stack(actions)-s['action_mean'])/s['action_std'],dtype=torch.float32,device=device),'action_is_pad':torch.tensor(np.stack(pads),dtype=torch.bool,device=device)}
        for k,v in images.items():
            x=torch.tensor(np.stack(v).transpose(0,3,1,2).copy(),dtype=torch.float32,device=device)/255
            batch[f'observation.images.{k}']=(x-torch.tensor([.485,.456,.406],device=device)[None,:,None,None])/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        return batch

def train(a):
    torch.set_num_threads(4);torch.manual_seed(731);np.random.seed(731);random.seed(731)
    paths=[p for p in sorted((ROOT/'datasets').glob('demo-*/demonstration.npz')) if json.loads((p.parent/'summary.json').read_text())['task_completed']]
    if len(paths)<8:raise RuntimeError('Need at least 8 successful demos')
    train_data=Data(paths[:-4]);validation=Data(paths[-4:],train_data.stats);config=make_config(a.chunk);config.optimizer_lr=a.lr;config.optimizer_lr_backbone=a.lr/10
    dest=ROOT/'checkpoints'/a.name;dest.mkdir(parents=True,exist_ok=False)
    model=ACTPolicy(config).to('mps');optimizer=torch.optim.AdamW(model.get_optim_params(),lr=a.lr,weight_decay=1e-4)
    if a.resume:
        old=torch.load(a.resume,map_location='cpu',weights_only=False);model.load_state_dict(old['model'])
    atomic_json(dest/'stats.json',train_data.stats);config.save_pretrained(dest)
    atomic_json(dest/'provenance.json',{'seed':731,'training':{str(p):sha(p) for p in paths[:-4]},'validation':{str(p):sha(p) for p in paths[-4:]},'not_used':'all final evaluation states','official_implementation':'lerobot.policies.act.modeling_act.ACTPolicy','image_preprocessing':'ImageNet normalization','expert':'scripted, calibrated model-based controller','batch':a.batch,'steps':a.steps,'resume':a.resume,'learning_rate':a.lr,'training_code_sha256':sha(__file__)})
    started=time.monotonic();log=(dest/'loss.jsonl').open('w');running=[]
    for step in range(1,a.steps+1):
        if (ROOT/'STOP').exists():raise InterruptedError('STOP')
        model.train();batch=train_data.batch(a.batch,a.chunk,'mps');loss,parts=model(batch);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();running.append(float(loss.detach().cpu()))
        if step%100==0 or step==1:
            model.eval()
            with torch.no_grad():
                batch=validation.batch(a.batch,a.chunk,'mps');pred=model.predict_action_chunk(batch);v=((pred-batch['action']).abs()*(~batch['action_is_pad']).unsqueeze(-1)).sum()/((~batch['action_is_pad']).sum()*16)
            record={'step':step,'train_loss':float(np.mean(running)),'validation_zero_latent_l1':float(v.cpu()),'wall_s':time.monotonic()-started,'parts':parts};running=[]
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
        if step%1000==0 or step==a.steps:
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'stats':train_data.stats,'chunk':a.chunk,'step':step,'config':config.to_dict() if hasattr(config,'to_dict') else None},dest/f'step-{step:06d}.pt')
    log.close();atomic_json(dest/'complete.json',{'steps':a.steps,'wall_s':time.monotonic()-started})

class Policy:
    def __init__(self,path):
        torch.set_num_threads(4);torch.manual_seed(731);self.path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=False);self.stats=saved['stats'];self.config=make_config(saved['chunk']);self.config.pretrained_backbone_weights=None;self.model=ACTPolicy(self.config).to('mps');self.model.load_state_dict(saved['model']);self.model.eval();self.calls=0
    def predict(self,inputs):
        s=self.stats;batch={'observation.state':torch.tensor((inputs['state']-s['state_mean'])/s['state_std'],dtype=torch.float32,device='mps')[None]}
        for k,v in inputs['images'].items():
            x=torch.tensor(v.transpose(2,0,1).copy(),dtype=torch.float32,device='mps')[None]/255
            batch[f'observation.images.{k}']=(x-torch.tensor([.485,.456,.406],device='mps')[None,:,None,None])/torch.tensor([.229,.224,.225],device='mps')[None,:,None,None]
        with torch.no_grad():actions=self.model.predict_action_chunk(batch)[0].cpu().numpy()
        self.calls+=1
        return actions*np.asarray(s['action_std'])+np.asarray(s['action_mean'])
