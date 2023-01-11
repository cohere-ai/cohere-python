from concurrent.futures import ThreadPoolExecutor
from typing import List
import tempfile
import os
import requests
import shutil

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
        return f'EmbedJob<id: {self.job_id}, status: {self.status}>'

    def __download_file(self, url):
        r = requests.get(url)
        temp = tempfile.NamedTemporaryFile(delete=True)
        temp.write(r.content)
        temp.seek(0)
        return temp

    def download_output(self, output_file=""):
        if output_file == "":
            output_file = f"{self.job_id}.jsonl"
        if self.status != 'complete':
            raise CohereError('job must be complete to download')
        with ThreadPoolExecutor() as exector:
            with open(output_file,'wb') as output:
                for temp_file in exector.map(self.__download_file, self.output_urls):
                    shutil.copyfileobj(temp_file, output)
                    temp_file.close()

