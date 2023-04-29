import pandas
import json
import sklearn
import nltk
import re
import math
from math import log
from collections import Counter, defaultdict
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)


metadata_csv = pandas.read_csv("new_dataset/movies_metadata.csv",
                               dtype={'fourth_column': 'str', 'sixth_column': 'int',
                                      'eight_column': 'str', 'eleventh_column': 'str',
                                      'twelfth_column': 'str', 'thirteenth_column': 'str',
                                      'fourteenth_column': 'int', 'fifteenth_column': 'int',
                                      'eigteenth_column': 'str', 'nineteenth_column': 'str'}, low_memory=False)


metadata_csv = metadata_csv.replace('1997-08-20', '0')
metadata_csv = metadata_csv.replace('2012-09-29', '0')
metadata_csv = metadata_csv.replace('2014-01-01', '0')

keywords_csv = pandas.read_csv("new_dataset/keywords.csv")
credits_csv = pandas.read_csv("new_dataset/credits.csv", dtype={'first_column': 'str', 'third_column': 'int'})

metadata_csv['id']=metadata_csv['id'].astype(int)

data_csv = pandas.merge(metadata_csv, keywords_csv, on='id', how='inner')
data_csv = pandas.merge(data_csv, credits_csv, on='id', how='inner')

data_csv.to_json(r'data_json.json')
data = pandas.read_json('data_json.json')

###Instance and Classification Classes

class MovieInstance:

    def __init__(
            self, id: str, title: str, studio: str, release_date: str, genre: str, synopsis: str, character_info: str, keywords: str, runtime: int, revenue: int
    ) -> None:
        self.id: str = id
        self.title: str = title
        self.studio: str = studio
        self.release_date: str = release_date
        self.genre: str = genre
        self.synopsis: str = synopsis
        self.character_info: str = character_info
        self.keywords:str = keywords
        self.runtime: int = runtime
        self.revenue: int = revenue

    def __repr__(self) -> str:
        return f"<MovieInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"title={self.title}; release_date={self.release_date}; genre ={self.genre}; synopsis={self.synopsis}"


class ClassificationInstance:

    def __init__(
            self, id: int, title: str, studio: tuple, release_season: str, genre: tuple, synopsis_ngrams: Iterable[str], character_match: int, keywords: list[str], runtime: int, revenue: int
    ) -> None:
        self.id: int = id
        self.title: str = title
        self.studio: tuple[str] = studio
        self.release_season: str = release_season,
        self.genre: tuple[str] = genre
        self.synopsis_ngrams: str = synopsis_ngrams
        self.character_match: int = character_match
        self.keywords: list[str] = keywords
        self.runtime: int = runtime
        self.revenue: int = revenue

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"title={self.title}; release_date={self.release_season}; genre = {self.genre}; synopsis={self.synopsis_ngrams}; runtime={self.runtime}; revenue={self.revenue}"

    def get_ngrams(self):
        return self.synopsis_ngrams

    def hide_label(self):
        self.label = "<>"


###Helper Functions

def get_season(release_date: str):
    try:
        month = int(release_date[5:7])
    except ValueError:
        return 'Q1'

    if month >= 10:
        return 'Q4'
    elif month >= 7:
        return 'Q3'
    elif month >= 4:
        return 'Q2'
    else:
        return 'Q1'


def read_from_json(raw_tag: str, column: str) -> list[str]:
    try:
        tags = []
        p = re.compile('(?<!\\\\)\'')
        p = re.compile('\'')
        raw_tag = p.sub('\"', raw_tag)

        structure = json.loads(raw_tag)
        for j_dict in structure:
            tags.extend(j_dict[column].lower().split(" "))
    except json.decoder.JSONDecodeError:
        pass
    return tags


def regex_from_struct_char(raw_tag: str) -> list[str]:
    characters = re.findall('(?<=\'character\': \').*?(?=\',)', raw_tag)
    return characters


def regex_from_struct_name(raw_tag: str) -> list[str]:
    items = re.findall('(?<=\'name\': \').*?(?=\',)', raw_tag)
    return items


def handle_runtime(n: float):
    if math.isnan(n):
        return -1
    else:
        return n

def revenue_label(rev: int):
    if rev > 50_000_000:
        return 'over'
    else:
        return 'under'

##Extract and Load
class FeatureExtractor:
    @staticmethod
    def extract_features(instance: MovieInstance) -> ClassificationInstance:
        studio = regex_from_struct_name(instance.studio)
        release_season = get_season(instance.release_date)
        genre_tag = regex_from_struct_name(instance.genre)
        instance.synopsis = instance.synopsis.lower()
        characters = regex_from_struct_char(instance.character_info)

        character_match = 0
        for character in characters:
            if character.lower() in instance.synopsis:
                character_match += 1

        synopsis_ngrams = instance.synopsis.split(" ") + regex_from_struct_name(instance.keywords) + characters

        keywords = regex_from_struct_name(instance.keywords)

        revenue = revenue_label(instance.revenue)
        return ClassificationInstance(instance.id, instance.title, studio, release_season, genre_tag, synopsis_ngrams, character_match, keywords, instance.runtime, revenue)


ext = FeatureExtractor()
def load_instances(start, stop):
    for line in range(start, stop):
        if data['revenue'][line] != 0:
            movie = MovieInstance(int(data['id'][line]), str(data['title'][line]), str(data['production_companies'][line]), str(data['release_date'][line]), str(data['genres'][line]),
                                str(data['overview'][line]) + str(data['tagline'][line]), str(data['cast'][line]), str(data['keywords'][line]), int(handle_runtime(data['runtime'][line])), data['revenue'][line])
            yield ext.extract_features(movie)