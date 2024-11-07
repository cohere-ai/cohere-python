

def lazy_sagemaker():
    try:
        import sagemaker as sage
        return sage
    except ImportError:
        raise CohereError("Sagemaker not available. Please install sagemaker.")

def lazy_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise CohereError("Boto3 not available. Please install lazy_boto3().")
    
def lazy_botocore():
    try:
        import botocore
        return botocore
    except ImportError:
        raise CohereError("Botocore not available. Please install botocore.")

