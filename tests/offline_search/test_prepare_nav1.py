import json
import pickle

from navsim.offline_search.prepare_nav1 import select_rows


def test_navtrain_selection_excludes_validation_logs_even_if_in_training(tmp_path):
    scene_dir = tmp_path / 'logs'
    scene_dir.mkdir()
    filter_dir = tmp_path / 'navsim/planning/script/config/common/train_test_split/scene_filter'
    train_dir = tmp_path / 'navsim/planning/script/config/training'
    filter_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    tokens = []
    for log in ['training_a', 'training_b', 'also_in_val']:
        frames = [{'token': f'{log}_{i}', 'log_name': log, 'roadblock_ids': ['road']} for i in range(20)]
        (scene_dir / f'{log}.pkl').write_bytes(pickle.dumps(frames))
        tokens.extend(f['token'] for f in frames)
    # JSON is valid YAML; avoid a second serializer in the test fixture.
    (filter_dir / 'navtrain.yaml').write_text(json.dumps({'tokens': tokens, 'log_names': ['training_a','training_b','also_in_val']}))
    (train_dir / 'default_train_val_test_log_split.yaml').write_text(json.dumps({
        'train_logs': ['training_a','training_b','also_in_val'], 'val_logs': ['also_in_val']}))
    rows = select_rows(tmp_path, scene_dir, 16, 123)
    assert len(rows) == 4
    assert {r['log_name'] for r in rows} == {'training_a', 'training_b'}
    assert rows == select_rows(tmp_path, scene_dir, 16, 123)
    assert all(r['metric_cache'] is None for r in rows)


def test_selection_supports_128_distinct_logs(tmp_path):
    scene_dir=tmp_path/'logs'; scene_dir.mkdir()
    filters=tmp_path/'navsim/planning/script/config/common/train_test_split/scene_filter'
    training=tmp_path/'navsim/planning/script/config/training'
    filters.mkdir(parents=True); training.mkdir(parents=True)
    logs=[f'log_{i}' for i in range(128)]; tokens=[]
    for log in logs:
        frames=[{'token':f'{log}_{i}', 'log_name':log, 'roadblock_ids':['road']} for i in range(14)]
        (scene_dir/f'{log}.pkl').write_bytes(pickle.dumps(frames)); tokens.append(frames[3]['token'])
    (filters/'navtrain.yaml').write_text(json.dumps({'tokens':tokens,'log_names':logs}))
    (training/'default_train_val_test_log_split.yaml').write_text(json.dumps({'train_logs':logs,'val_logs':[]}))
    rows=select_rows(tmp_path,scene_dir,128,123)
    assert len(rows)==128 and len({r['log_name'] for r in rows})==128
