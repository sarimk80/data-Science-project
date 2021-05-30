from dagster import pipeline
from dagster_dbt import dbt_cli_run

config = {"project-dir": "/yolov5/main.py"} 
run_all_models = dbt_cli_run.configured(config, name="run_dbt_project")

@pipeline
def my_dbt_pipeline():
    run_all_models()#fine? have you installed dagster? i dont think so
    
    #going for prayer abhi aae bus ok