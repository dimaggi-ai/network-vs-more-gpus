import copy
import unittest
from validate_evidence import validate


class ManifestTests(unittest.TestCase):
    def test_mutations(self):
        row = dict(name='pue',value=1.2,unit='dimensionless',evidence_class='scenario',sources=['s'],
                   transformation='identity',uncertainty='hypothetical',affected_outputs=['power'],valid_bounds=[1,5])
        d = dict(schema_version='dimaggi-evidence/v1',inputs=[row],results=[],sources={'s':dict(
            source_type='model-assumption',location=__file__,version_date='2026-09-10',
            license_or_reuse='original',scope='hypothetical envelope')})
        self.assertEqual(validate(d,'.'),[])
        for field,value in [('unit',''),('value',float('nan')),('value',.5),('sources',['removed']),('evidence_class','empirical-ish'),
                            ('valid_bounds',[float('nan'),5]),('valid_bounds',[1,float('inf')]),
                            ('valid_bounds',['1',5]),('valid_bounds',[5,1])]:
            changed=copy.deepcopy(d);changed['inputs'][0][field]=value
            self.assertTrue(validate(changed,'.'),field)


if __name__ == '__main__': unittest.main()
