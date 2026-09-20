"""Read-only P2 data preflight; never load a model, weights, or held-out labels."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_contract import load_contract,validate_datastore

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',required=True)
    p.add_argument('--members',required=True)
    p.add_argument('--datastore-build',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    _,members=load_contract(args.protocol,args.members)
    result=validate_datastore(members,args.datastore_build)
    with Path(args.output).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
