
warning = "AWS dependencies are not installed. Please install boto3, botocore, and sagemaker."

def lazy_sagemaker():
    try:
        import sagemaker as sage # type: ignore
        return sage
    except ImportError:
        raise ImportError(warning)

def lazy_boto3():
    try:
        import boto3 # type: ignore
        return boto3
    except ImportError:
        raise ImportError(warning)
    
def lazy_botocore():
    try:
        import botocore # type: ignore
        return botocore
    except ImportError:
        raise ImportError(warning)

