from src.run_pipeline import run
def test_pipeline_smoke(tmp_path):
    # The pipeline accepts a small explicit smoke dataset while production defaults remain >=100k.
    result=run(5000,seed=9); assert result['features']==23 and result['n_customers']==5000
