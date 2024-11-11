

def lazy_sagemaker():
    try:
        import sagemaker as sage # type: ignore
        return sage
    except ImportError:
        raise ImportError("Sagemaker not available. Please install sagemaker.")

def lazy_boto3():
    try:
        import boto3 # type: ignore
        return boto3
    except ImportError:
        raise ImportError("Boto3 not available. Please install lazy_boto3().")
    
def lazy_botocore():
    try:
        import botocore # type: ignore
        return botocore
    except ImportError:
        raise ImportError("Botocore not available. Please install botocore.")

