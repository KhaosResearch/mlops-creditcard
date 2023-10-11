from prefect.blocks.kubernetes import KubernetesClusterConfig
from prefect.filesystems import S3
from prefect.infrastructure.kubernetes import KubernetesJob

cluster_config_block = KubernetesClusterConfig.load("k8s-config")
s3_block = S3.load("khaos-minio")

environment = {
        'PREFECT_API_URL': 'http://192.168.219.71:32000/api',
        'FSSPEC_S3_ENDPOINT_URL': 'http://192.168.219.2:9000',
        'AWS_ACCESS_KEY_ID': s3_block.aws_access_key_id.get_secret_value(),
        'AWS_SECRET_ACCESS_KEY': s3_block.aws_secret_access_key.get_secret_value(),
        'MLFLOW_TRACKING_URI': 'http://192.168.219.71:32001',
        'MLFLOW_S3_ENDPOINT_URL': "http://192.168.219.2:9000",
        'MLFLOW_S3_IGNORE_TLS': "true"
    }

customizations = [
    {
        "op": "add",
        "path": "/spec/template/spec/containers/0/resources",
        "value": {
            "requests": {
                "cpu": "2",
                "memory": "16Gi"
            },
            "limits": {
                "cpu": "2",
                "memory": "16Gi"
            }
        },
    },{
        "op": "replace",
        "path": "/spec/template/spec/parallelism",
        "value": 1,
    },{
        "op": "remove",
        "path": "/spec/template/spec/completions"
    }
]

infra_k8s = KubernetesJob(
    env=environment,
    image="ghcr.io/khaosresearch/prefect-landcover:latest",
    namespace="mlops-prefect",
    image_pull_policy="Always",
    cluster_config=cluster_config_block,
    job=KubernetesJob.base_job_manifest(),
    customizations=customizations,
    pod_watch_timeout_seconds=300,
    finished_job_ttl=600
)

infra_k8s.save("k8s-infra-retraining-creditcard", overwrite=True)