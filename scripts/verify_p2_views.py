"""Read-only real-data view/scaler/collation probe; never constructs a model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from p2_contract import load_contract, PANELS, ROUTES
from p2_data import P2Data, schedule


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','members','datastore','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    _,members=load_contract(a.protocol,a.members)
    data=P2Data(members,a.datastore)
    try:
        views={f'{panel}_{route}':data.source(panel,route) for panel in PANELS for route in ROUTES}
        views['target']=data.target();result={}
        for key,view in views.items():
            probes=[]
            for split in ('train','validation'):
                for task,rows in view.rows[split].items():
                    batch,selected=view.batch(split,task,[0],'cpu')
                    if batch.y.shape!=(1,1) or float(batch.y[0,0])!=selected[0]['label']:
                        raise ValueError('collated label mismatch')
                    probes.append(dict(split=split,task=task,sample_id=selected[0]['sample_id'],label=selected[0]['label']))
            result[key]=dict(identity=view.identity,scalers=view.scalers,
                counts={s:{t:len(rr) for t,rr in table.items()} for s,table in view.rows.items()},
                updates_per_epoch=len(schedule(view.tasks,{t:len(view.rows['train'][t]) for t in view.tasks},42,0)),probes=probes)
        with Path(a.output).open('x',encoding='utf-8') as f:
            json.dump(dict(status='PASS',scope='read_only_views_no_model_no_test_no_calibration',views=result),f,ensure_ascii=False,indent=2,allow_nan=False)
        print('PASS: four source views and Human3 train/validation; no model forward')
    finally:data.close()


if __name__=='__main__':main()
