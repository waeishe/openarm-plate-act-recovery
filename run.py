"""Entry point for the portable submission. Simulator and outputs stay in this folder."""
import os,sys,argparse,json,hashlib,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('TORCH_HOME',str(ROOT/'cache/torch'));os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK','1')

def lines(path):
    if path.exists():return [json.loads(s) for s in path.read_text().splitlines()]
    with gzip.open(str(path)+'.gz','rt') as f:return [json.loads(s) for s in f]

def score(directory):
    from environment import evaluate,opening_events
    h=lines(directory/'evaluation.jsonl');a=lines(directory/'actions.jsonl');old=json.loads((directory/'summary.json').read_text());new=evaluate(h,opening_events(a),old['error'])
    for k,v in new.items():
        if k=='final':continue
        if isinstance(v,float):assert abs(old[k]-v)<1e-8,(directory,k)
        else:assert old[k]==v,(directory,k)
    return new

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify');sub.add_parser('rescore')
    d=sub.add_parser('eval');d.add_argument('--run',required=True);d.add_argument('--method',choices=['act','agent'],default='act');d.add_argument('--checkpoint',default=str(ROOT/'checkpoints/plate-act.pt'));d.add_argument('--recorded-decisions',choices=[f'pair-{i:02d}-agent' for i in range(1,6)],help='Offline validation ONLY; replay saved requests, do not call a model.')
    d=sub.add_parser('demos');d.add_argument('--count',type=int,default=24)
    d=sub.add_parser('train');d.add_argument('--name',required=True);d.add_argument('--steps',type=int,required=True);d.add_argument('--batch',type=int,default=16);d.add_argument('--resume');d.add_argument('--chunk',type=int,default=40);d.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args();protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    if a.command=='verify':
        from agent_control import PROMPT
        assert hashlib.sha256((ROOT/'checkpoints/plate-act.pt').read_bytes()).hexdigest()==protocol['checkpoint_sha256']
        assert hashlib.sha256(PROMPT.encode()).hexdigest()==protocol['agent_prompt_sha256']
        print(json.dumps({'checkpoint':'matched original experiment','agent_prompt':'matched original experiment'}));return
    if a.command=='rescore':
        from environment import evaluate,opening_events
        import statistics
        original={x['run']:x for x in json.loads((ROOT/'evidence/results.json').read_text())};result={}
        for d in sorted((ROOT/'evidence/trials').iterdir()):
            s=score(d);old=original[d.name];entry={'completed':s['task_completed'],'quality':s['quality_success']}
            assert entry['completed']==old['task_completed'] and entry['quality']==old['quality_success']
            if old['method']=='agent':
                end=json.loads((d/'act-end.json').read_text());handoff=end['sim_time'];h=lines(d/'evaluation.jsonl')
                post=evaluate([x for x in h if x['sim_time']>=handoff-1e-8],[t for t in s['release_times'] if t>=handoff])
                assert post['quality_success']==old['post_handoff_quality']
                entry['post_handoff_quality']=post['quality_success']
                if s['task_completed']:
                    completion=s['terminal_stable_completion_s'];sim=completion-handoff
                    wall=next(x['wall_s'] for x in lines(d/'timing.jsonl') if x['sim_time']>=completion-1e-8)-end['wall_s']
                    assert abs(sim-old['recovery_sim_s'])<1e-8 and abs(wall-old['recovery_wall_to_stable_s'])<1e-8
                    entry.update(recovery_sim_s=sim,recovery_wall_s=wall)
            result[d.name]=entry
        success=[r for r in result.values() if 'recovery_sim_s' in r]
        print(json.dumps({'trials':result,'success_count':len(success),'median_recovery_sim_s':statistics.median(r['recovery_sim_s'] for r in success),'median_recovery_wall_s':statistics.median(r['recovery_wall_s'] for r in success)},indent=2));return
    if a.command=='demos':
        from demonstrations import demos
        demos(a);return
    if a.command=='train':
        from learning import train
        train(a);return
    if not a.run.replace('-','').replace('_','').isalnum():raise ValueError('Use a unique alphanumeric run name')
    if a.recorded_decisions:
        if a.method!='agent':raise ValueError('Recorded-decision replay requires --method agent')
        from openarm_recovery import agent
        source=ROOT/'evidence/trials'/a.recorded_decisions/'recorded-decisions';counter=[0]
        def replay(directory,packet,images,config,stop,timeout=180):
            index=counter[0];counter[0]+=1;f=source/f'decision-{index:03d}.json'
            value=json.loads(f.read_text());directory.mkdir(parents=True);(directory/'packet.json').write_text(json.dumps(packet));(directory/'response.json').write_text(json.dumps(value));(directory/'REPLAY_ONLY.txt').write_text('Recorded response playback. No model was called. Not a new experimental replicate.')
            return value,{'total_tokens':0,'wall_seconds':0,'replayed':True}
        agent.decide=replay
    a.x_mm=a.y_mm=a.source_y_mm=0.;a.source_x_mm=5.;a.horizon=protocol['horizon'];a.handoff=protocol['handoff'];a.execute=protocol['execute'];a.agent_config=protocol['agent_settings']
    from evaluation import run
    run(a)
if __name__=='__main__':main()
