import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import json

import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
from spacy.matcher import Matcher

def convect_skills_from_txt_to_Jsonl():
    """Convert skills (txt format) to patterns (jsonl format)"""

    file_txt = open("../data/skills_in_demand.txt", "r", encoding='ISO-8859-1') # iso encoding to handle non-ascii characters
    list_skills_in_demand = []
    for x in file_txt.readlines():
        list_skills_in_demand.append(x.strip())
    file_txt.close()

    # Create the skill patterns
    rule_patterns = []
    for skill in list_skills_in_demand:
        pattren = []
        for elt in skill.split(" "):
            pattren.append({"LOWER": elt})

        json_data = {"label": "SKILL", "pattern": pattren}
        # Convert the dictionary to a JSON string with double quotes
        json_string = json.dumps(json_data, ensure_ascii=True)
        rule_patterns.append(json_string)

    # Save patterns to jsonl file
    file_jsonl = open("../data/Skill_patterns.jsonl", "w")
    for k in range(len(rule_patterns)):
        file_jsonl.write(rule_patterns[k] + "\n")
    file_jsonl.close()

"""
This function creates a spaCy NLP pipeline that includes an Entity Ruler 
to recognize skill-related entities from text. It loads a pre-trained 
language model, adds rule-based entity recognition, and disables NER to avoid conflicts.
"""
def Spacy_create_nlp():
    # Disables the built-in Named Entity Recognition (NER) because some skills (like "SQL", "NoSQL") 
    # might be incorrectly recognized as organizations instead of skills.
    nlp = spacy.load("en_core_web_lg", disable=["ner"])
    # nlp = spacy.load("en_core_web_lg")

    print("Create Spacy ruler...\n\n")

    convect_skills_from_txt_to_Jsonl()
    skill_pattern_path = "../data/skill_patterns.jsonl"

    ruler = nlp.add_pipe("entity_ruler")
    ruler.from_disk(skill_pattern_path)
    # print(nlp.pipe_names, "\n\n")

    return nlp


# create columns: skills, missing skills and match_score
def get_skills(nlp, text):
    doc = nlp(text)
    list_skills = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            list_skills.append(ent.text.lower())
    return list_skills


def unique_skills(list_skills):
    return list(set(list_skills))


def get_match_score(job_skills, your_skills):
    """get a percentage of match skills.
    Inputs:
    - Job_skills (list): list of required skills.
    - your_skills (str): your skills (comma separated)
    """
    # convert your skills to list
    list_your_skills = your_skills.split(",")
    score = 0
    for x in job_skills:
        if x in list_your_skills:
            score += 1
    job_skills_len = len(job_skills)
    try:
        match_score = round(score / job_skills_len * 100, 1)
    except:
        match_score = None
    return match_score


def get_missing_skills(list_job_skills, str_your_skills, return_list=True):
    list_your_skills = str_your_skills.split(",")
    list_your_skills = [str.lower(skill) for skill in list_your_skills]

    missing_skills = []

    for x in list_job_skills:
        if str.lower(x) not in list_your_skills:
            try:
                missing_skills.append(x)
            except:
                pass
    if return_list:
        return missing_skills
    else:
        return ",".join(missing_skills)


def update_LinkedinJobs_DF(df, your_skills):
    # convert list to str to avoid error in np.vectorize
    your_skills = ",".join(your_skills)

    # create/update match score and missing skills columns
    df["match_score"] = np.vectorize(get_match_score)(df["skills"], your_skills)
    df["missing_skills"] = np.vectorize(get_missing_skills)(
        df["skills"], your_skills, return_list=False
    )
    df = df.sort_values(by="match_score", ascending=False)

    # Saving data to json
    df.to_json("../data/linkedin_jobs_scraped.json")

    return df


# Display the job and highlight the skills user do and user do not have
from spacy.matcher import Matcher


def get_pattern(skill, rule_based_matching="LOWER"):
    skill_split = skill.split()  # example: 'data science' --> ['data', 'science']
    pattern = []
    for obj in skill_split:
        pattern.append({rule_based_matching: obj})
    return pattern


def get_matchers(list_skills, missing_skills, spacy_nlp):
    list_missing_skills = missing_skills
    matcher_OK = Matcher(spacy_nlp.vocab)  # you have this skill
    matcher_NOK = Matcher(spacy_nlp.vocab)  # you do not have this skill

    for k, skill in enumerate(list_skills):
        # print(k, skill)
        pattern = get_pattern(skill)
        # print(k, pattern)

        if skill in list_missing_skills:
            matcher_NOK.add(f"rule_{k}", [pattern])  # you do not have this skill
        else:
            matcher_OK.add(f"rule_{k}", [pattern])  # you have this skill

    return matcher_OK, matcher_NOK


def get_indexes(list_matches):
    """For each match, the output has three elements.
    - The first element is the match ID.
    - The second and third elements are the positions of the matched tokens."""
    list_indexes = []
    for match in list_matches:
        start_index = match[1]
        end_index = match[2]
        for k in range(start_index, end_index):
            list_indexes.append(k)
    return list_indexes


def return_words_types(job_txt, list_required_skills, list_missing_skills, spacy_nlp):
    """Returns two lists:
    - list of words
    - list of the corresponding types (skill, missing skill, other words)"""

    Job_doc_nlp = spacy_nlp(job_txt)

    matcher_OK, matcher_NOK = get_matchers(
        list_required_skills, list_missing_skills, spacy_nlp
    )

    list_matches_OK = matcher_OK(Job_doc_nlp)
    list_matches_NOK = matcher_NOK(Job_doc_nlp)

    indexes_SKILLS_OK = get_indexes(list_matches_OK)
    indexes_SKILLS_NOK = get_indexes(list_matches_NOK)

    indexes_others = []

    for k, word in enumerate(Job_doc_nlp):
        if (k in indexes_SKILLS_OK) | (k in indexes_SKILLS_NOK):
            pass
        else:
            indexes_others.append(k)

    words = []
    types = []

    for k, word in enumerate(Job_doc_nlp):
        words.append(word)
        if k in indexes_SKILLS_NOK:
            types.append("SKILL-missing")
        elif k in indexes_SKILLS_OK:
            types.append("SKILL")
        else:
            types.append("other")

    return words, types