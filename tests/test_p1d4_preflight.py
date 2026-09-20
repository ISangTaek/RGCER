from pathlib import Path
import json
import xml.etree.ElementTree as ET
import pytest
from scripts.p1d4_preflight import check_wsl,sha,REPO
from p1d_optimization import OptimizationError


def fixture(root,*,skip=False,missing=False):
    tree=ET.Element('testsuites');suite=ET.SubElement(tree,'testsuite')
    files=sorted((REPO/'tests').glob('test_p1d*.py'))
    for path in files[:-1] if missing else files:
        case=ET.SubElement(suite,'testcase',classname='tests.'+path.stem,name='synthetic_receipt_test')
        if skip:ET.SubElement(case,'skipped')
    ET.ElementTree(tree).write(root/'tests.xml')
    (root/'preflight.json').write_text(json.dumps(dict(role='wsl',commit='a'*40,exit_code=0,junit_sha256=sha(root/'tests.xml'))))


def test_wsl_requires_raw_complete_non_skipped_suite(tmp_path):
    fixture(tmp_path)
    assert check_wsl(tmp_path,'a'*40)['exit_code']==0


@pytest.mark.parametrize('fault',['skip','missing','wrong_commit','changed_xml'])
def test_receipt_pass_cannot_hide_missing_or_wrong_tests(tmp_path,fault):
    fixture(tmp_path,skip=fault=='skip',missing=fault=='missing')
    if fault=='changed_xml':(tmp_path/'tests.xml').write_text('<testsuites/>')
    with pytest.raises(OptimizationError):check_wsl(tmp_path,'b'*40 if fault=='wrong_commit' else 'a'*40)
