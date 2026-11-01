"""Actual image-conditioned model calls. Prompt is set by agent_control."""
import json,os,signal,subprocess,time
from pathlib import Path
from aspire_lab.isolated import codex_args,parse_events
from .runtime import atomic_json,digest,now,read
API=""
def settings():
    return {"model":"gpt-6-astra","effort":"medium"}

def decide(directory, packet, images, config, stop, timeout=180):
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory/'packet.json', packet)
    atomic_json(directory/'schema.json', {'type':'object', 'properties': {k:{'type':'string'} for k in ('reason','request_json')}, 'required':['reason','request_json'], 'additionalProperties':False})
    prompt = API + '\nINPUT:\n' + json.dumps(packet, ensure_ascii=False)
    (directory/'prompt.txt').write_text(prompt)
    args = codex_args(directory, config['model'], config['effort'])
    for im in images:
        args.extend(['--image', str(im)])
    args.append('-')
    atomic_json(directory/'invocation.json', {'argv':args, 'started_at':now(), 'images':[{'path':str(im),'sha256':digest(im)} for im in images], 'packet_sha256':digest(directory/'packet.json')})
    started = time.monotonic()
    with (directory/'events.jsonl').open('w') as out, (directory/'stderr.log').open('w') as err:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, start_new_session=True)
        try:
            process.stdin.write(prompt)
            process.stdin.close()
            while process.poll() is None:
                if stop.exists():
                    raise InterruptedError('user_STOP')
                if time.monotonic()-started > timeout:
                    raise TimeoutError('agent_decision_timeout')
                time.sleep(.2)
            if process.returncode:
                raise RuntimeError(f'agent_exit_{process.returncode}: see {directory}/stderr.log')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    accounting = parse_events([json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines() if s.strip()])
    accounting['wall_seconds'] = time.monotonic()-started
    atomic_json(directory/'accounting.json', accounting)
    return read(directory/'response.json'), accounting
