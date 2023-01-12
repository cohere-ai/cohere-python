from concurrent.futures import ThreadPoolExecutor, Future
from typing import List
import tempfile
import os
import requests
import shutil
import time
from tqdm import tqdm

from cohere.response import AsyncAttribute, CohereObject
from cohere.error import CohereError

class EmbedJob(CohereObject):


    def __init__(self, job_id: str, status: str, created_at, input_url: str, output_urls: List[str], model: str, truncate: str, percent_complete: float) -> None:
        self.job_id = job_id
        self.status = status
        self.created_at = created_at
        self.input_url = input_url
        self.output_urls = output_urls
        self.model = model
        self.truncate = truncate
        self.percent_complete = percent_complete

    def __repr__(self) -> str:
        if self.percent_complete > 0:
            return f'EmbedJob<id: {self.job_id}, status: {self.status} : {self.percent_complete}%>'
        return f'EmbedJob<id: {self.job_id}, status: {self.status}>'

    def __download_file(self, url):
        response = requests.get(url, stream=True)
        temp = tempfile.NamedTemporaryFile(delete=True)
        with tqdm.wrapattr(open(os.devnull, "wb"), "write", miniters=1, total=int(response.headers.get('content-length', 0))) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)
                temp.write(chunk)
        temp.seek(0)
        return temp

    def download_output(self, output_file=""):
        if self.status != 'complete':
            raise CohereError('job must be complete to download')
        if output_file == "":
            output_file = f"{self.job_id}.jsonl"
        with ThreadPoolExecutor() as exector:
            with open(output_file,'wb') as output:
                for temp_file in exector.map(self.__download_file, self.output_urls):
                    try:
                        shutil.copyfileobj(temp_file, output)
                    finally:
                        temp_file.close()
