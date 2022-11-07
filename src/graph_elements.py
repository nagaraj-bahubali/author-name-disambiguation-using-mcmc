"""
This module defines basic building blocks of the graph.
"""

from typing import List


class Paper:

    def __init__(self, p_id: int, title: str, co_authors: List[str], journal: str, year: int, author_class: int):
        self.p_id = p_id
        self.title = title
        self.co_authors = co_authors
        self.journal = journal
        self.year = year
        self.author_class = author_class

    def get_p_id(self):
        return self.p_id

    def get_title(self):
        return self.title

    def get_co_authors(self):
        return self.co_authors

    def get_year(self):
        return self.year

    def get_journal(self):
        return self.journal

    # def get_author_class(self):
    #     return self.author_class
    #
    # def get_info(self):
    #     return str(self.author_class) + " " + str(self.p_id)
        # return str(self.author_class) + " " + str(self.p_id) + " " + self.title + " " + str(self.co_authors) + " " + str(self.year)


class Graphlet:

    def __init__(self, g_id: int, atomic_name: str, papers: List[Paper]):
        self.g_id = g_id
        self.atomic_name = atomic_name
        self.papers = papers

    def get_g_id(self):
        return self.g_id

    def get_atomic_name(self):
        return self.atomic_name

    def get_papers(self) -> List[Paper]:
        return self.papers
