# pipeline-api
from enum import Enum
import re
import signal

from unstructured.staging.base import convert_to_isd
from prepline_sec_filings.sections import (
    ALL_SECTIONS,
    SECTIONS_10K
)

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#from pipeline-sec-filings import prepline_sec_filings

from prepline_sec_filings.sections import section_string_to_enum, validate_section_names, SECSection
from prepline_sec_filings.sec_document import SECDocument, REPORT_TYPES, VALID_FILING_TYPES
from prepline_sec_filings.fetch import *

import io

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        try:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        except ValueError:
            pass
    def __exit__(self, type, value, traceback):
        try:
            signal.alarm(0)
        except ValueError:
            pass

# pipeline-api
def get_regex_enum(section_regex):
    class CustomSECSection(Enum):
        CUSTOM = re.compile(section_regex)

        @property
        def pattern(self):
            return self.value

    return CustomSECSection.CUSTOM

# pipeline-api
import csv
from typing import Dict
from unstructured.documents.elements import Text, NarrativeText, Title, ListItem
def convert_to_isd_csv(results:dict) -> str:
    """
    Returns the representation of document elements as an Initial Structured Document (ISD)
    in CSV Format.
    """
    csv_fieldnames: List[str] = ["section", "element_type", "text"]
    new_rows = []
    for section, section_narrative in results.items():
        rows: List[Dict[str, str]] = convert_to_isd(section_narrative)
        for row in rows:
            new_row_item = dict()
            new_row_item["section"] = section
            new_row_item["element_type"] = row["type"]
            new_row_item["text"] = row["text"]
            new_rows.append(new_row_item)

    with io.StringIO() as buffer:
        csv_writer = csv.DictWriter(buffer, fieldnames=csv_fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(new_rows)
        return buffer.getvalue()

# pipeline-api
from unstructured.staging.label_studio import stage_for_label_studio

# List of valid response schemas
LABELSTUDIO = "labelstudio"
ISD = "isd"

def pipeline_api(text, response_type="application/json", response_schema="isd", m_section=[], m_section_regex=[]):
    """Many supported sections including: RISK_FACTORS, MANAGEMENT_DISCUSSION, and many more"""
    validate_section_names(m_section)

    sec_document = SECDocument.from_string(text)
    if sec_document.filing_type not in VALID_FILING_TYPES:
        raise ValueError(
            f"SEC document filing type {sec_document.filing_type} is not supported, "
            f"must be one of {','.join(VALID_FILING_TYPES)}"
        )
    results = {}
    if m_section == [ALL_SECTIONS]:
        filing_type = sec_document.filing_type
        if filing_type in REPORT_TYPES:
            if filing_type.startswith("10-K"):
                m_section = [enum.name for enum in SECTIONS_10K]
            elif filing_type.startswith("10-Q"):
                m_section = [enum.name for enum in SECTIONS_10Q]
            else:
                raise ValueError(f"Invalid report type: {filing_type}")

        else:
            m_section = [enum.name for enum in SECTIONS_S1]
    for section in m_section:
        results[section] = sec_document.get_section_narrative(
            section_string_to_enum[section]
        )
    for i, section_regex in enumerate(m_section_regex):
        regex_enum = get_regex_enum(section_regex)
        with timeout(seconds=5):
            section_elements = sec_document.get_section_narrative(regex_enum)
            results[f"REGEX_{i}"] = section_elements
    if response_type == "application/json":
        if response_schema == LABELSTUDIO:
            return {section:stage_for_label_studio(section_narrative) for section, section_narrative in results.items()}
        elif response_schema == ISD:
            return {section:convert_to_isd(section_narrative) for section, section_narrative in results.items()}
        else:
            raise ValueError(f"output_schema '{response_schema}' is not supported for {response_type}")
    elif response_type == "text/csv":
        if response_schema != ISD:
            raise ValueError(f"output_schema '{response_schema}' is not supported for {response_type}")
        return convert_to_isd_csv(results)
    else:
        raise ValueError(f"response_type '{response_type}' is not supported")

def _get_session(company: Optional[str] = None, email: Optional[str] = None) -> requests.Session:
    """Creates a requests sessions with the appropriate headers set. If these headers are not
    set, SEC will reject your request.
    ref: https://www.sec.gov/os/accessing-edgar-data"""
    if company is None:
        company = os.environ.get("SEC_API_ORGANIZATION")
    if email is None:
        email = os.environ.get("SEC_API_EMAIL")
    assert company
    assert email
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{company} {email}",
            "Content-Type": "text/html",
        }
    )
    return session

session = _get_session("IITM", "21f1001906@ds.study.iitm.ac.in")

def _drop_dashes(accession_number: Union[str, int]) -> str:
    """Converts the accession number to the no dash representation."""
    accession_number = str(accession_number).replace("-", "")
    return accession_number.zfill(18)

from datetime import datetime
from bs4 import BeautifulSoup

def get_10k_filings_by_ticker_with_years(ticker: str,
                                         company: Optional[str] = "IITM",
                                         email: Optional[str] = "21f1001906@ds.study.iitm.ac.in") -> List[Tuple[int, str]]:
    session = _get_session(company, email)
    cik = get_cik_by_ticker(session, ticker)
    forms_dict = get_forms_by_cik(session, cik)
    ten_k_filings = []
    for accession_number, form_type in forms_dict.items():
        if form_type == "10-K":
            text = get_filing(cik, _drop_dashes(accession_number), company, email)
            year = extract_filing_year(text)
            ten_k_filings.append((year, text))
    return ten_k_filings

def extract_filing_year(text: str) -> int:
    # Extract the filing year from the text content of the filing
    pattern = re.compile(r"CONFORMED PERIOD OF REPORT:\s*(\d{4})\d{4}")
    match = pattern.search(text)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Unable to extract filing year")

# Example usage
# ticker = "RGLD"  # Example ticker symbol for Apple Inc.
# ten_k_filings_with_years = get_10k_filings_by_ticker_with_years(ticker)

# for year, filing_text in ten_k_filings_with_years:
#     if int(year) < 2020:
#       continue
#     print(f"Year: {year}")
#     all_narratives = pipeline_api(filing_text, response_type="text/csv", m_section=["_ALL"])

#     with open(f'{ticker}/{year}.csv', 'w') as out:
#         out.write(all_narratives)
#     #print(f"Filing Text: {filing_text[:5000]}...")
#     #print("=" * 50)
