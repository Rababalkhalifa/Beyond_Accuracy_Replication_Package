import math
import unittest
import verify_rq3 as v


def row(gold,pred,usable=True,abstain=False,confidence=.8):
    return dict(gold=gold,pred=pred,usable=usable,abstain=abstain,confidence=confidence,
        latency=1. if usable else 0.,request_failed=not usable,status='ok' if usable else 'request_error',
        raw_status='ok' if usable else 'json_error',raw_saved_disagreement=False)


class MetricTests(unittest.TestCase):
    def test_macro_f1_known_confusion(self):
        self.assertAlmostEqual(v.macro_f1(['NEG','POS'],['NEG','NEG'],['NEG','POS']),1/3)

    def test_failure_fallback_is_not_a_model_success(self):
        scores=v.condition_scores([row('NEG','NEG',False),row('POS','POS')],['NEG','POS'])
        self.assertEqual(scores['archived_accuracy'],1.)
        self.assertEqual(scores['failure_aware_accuracy'],.5)
        self.assertEqual(scores['n_failure_fallback_label_correct'],1)
        self.assertEqual(scores['coverage_all_attempts'],.5)
        self.assertEqual(scores['archived_coverage'],1.)

    def test_zero_coverage_risk_stays_undefined(self):
        scores=v.condition_scores([row('NEG','NEG',abstain=True),row('POS','POS',abstain=True)],['NEG','POS'])
        self.assertIsNone(scores['retained_risk'])
        self.assertEqual(scores['coverage_all_attempts'],0)
        self.assertEqual(scores['abstain_rate_all_attempts'],1.)

    def test_no_abstentions_subgroup_error_undefined(self):
        scores=v.condition_scores([row('NEG','NEG')],['NEG','POS'])
        self.assertIsNone(scores['abstained_best_guess_risk'])

    def test_ece_endpoints(self):
        self.assertEqual(v.ece(['NEG','POS'],['POS','POS'],[0.,1.]),0.)

    def test_ece_not_an_accuracy_complement(self):
        self.assertAlmostEqual(v.ece(['NEG','POS'],['POS','POS'],[.9,.9]),.4)

    def test_json_boolean_confidence_rejected(self):
        status,_=v.inspect_raw('{"label":"POS","confidence":true,"abstain":false}',('NEG','POS'))
        self.assertEqual(status,'schema_error')

    def test_json_duplicate_and_nan_rejected(self):
        for raw in ['{"label":"POS","label":"NEG","confidence":0.9,"abstain":false}',
                    '{"label":"POS","confidence":NaN,"abstain":false}']:
            self.assertEqual(v.inspect_raw(raw,('NEG','POS'))[0],'json_error')

    def test_no_silent_markdown_cleanup(self):
        self.assertEqual(v.inspect_raw('```json\n{"label":"POS","confidence":0.9,"abstain":false}\n```',('NEG','POS'))[0],'json_error')

    def test_schema_extra_fields_and_wrong_domain(self):
        for raw in ['{"label":"POS","confidence":0.9,"abstain":false,"reason":"x"}',
                    '{"label":"BELIEVER","confidence":0.9,"abstain":false}']:
            self.assertEqual(v.inspect_raw(raw,('NEG','POS'))[0],'schema_error')

    def test_timing_quantile(self):
        self.assertAlmostEqual(v.quantile([0.,1.,2.,3.],.95),2.85)

    def test_bad_stored_boolean_is_not_coerced(self):
        with self.assertRaises(ValueError):v.boolean('maybe')

    def test_prompt_alignment_mismatch_is_rejected(self):
        conditions={}
        for m in v.MODELS:
            for p in v.PROMPTS:
                r=row('NEG','NEG');r.update(id='same',hash='h')
                if m==v.MODELS[1] and p==v.PROMPTS[2]:r['id']='different'
                conditions[v.SETTINGS[0],m,v.SEEDS[0],p]=[r]
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            v.prompt_comparisons(conditions,[])

    def test_average_within_seed_range_not_range_of_means(self):
        conditions={};scores=[]
        for setting in v.SETTINGS:
            for seed in v.SEEDS:
                for model in v.MODELS:
                    for pi,p in enumerate(v.PROMPTS):
                        rows=[]
                        for i,g in enumerate(['NEG','POS']):
                            r=row(g,g);r.update(id=str(i),hash='h'+str(i));rows.append(r)
                        conditions[setting,model,seed,p]=rows
                        value=([.9,.5,.5] if seed==42 else [.5,.9,.5] if seed==43 else [.7,.7,.7])[pi]
                        scores.append(dict(setting=setting,model=model,seed=seed,prompt=p,
                            archived_macro_f1=value,failure_aware_macro_f1=value,usable_macro_f1=value))
        _,summary,_,_,_=v.prompt_comparisons(conditions,scores)
        r=next(r for r in summary if r['setting']==v.SETTINGS[0] and r['model']==v.MODELS[0] and r['score']=='archived_macro_f1')
        self.assertAlmostEqual(r['mean_seed_range'],.16)


if __name__=='__main__':unittest.main()
