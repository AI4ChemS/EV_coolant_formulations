import time
import re
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


# ----------------------------
# Default H-phrases
# ----------------------------
DEFAULT_H_PHRASES = [
    "H200", "H201", "H202", "H203", "H204", "H205", "H206", "H207", "H208", "H209", "H210", "H211", "H220", "H221", "H222", "H223",
    "H224", "H225", "H226", "H228", "H229", "H230", "H231", "H232", "H240", "H241",
    "H242", "H250", "H251", "H252", "H260", "H261", "H270", "H271", "H272", "H280",
    "H281", "H282", "H283", "H284", "H290", "H300", "H301", "H302", "H303", "H304", "H305", "H310", "H311",
    "H312", "H313", "H314", "H315", "H316", "H317", "H318", "H319", "H320", "H330",
    "H331", "H332", "H333", "H334", "H335", "H336", "H340", "H341", "H350", "H351",
    "H360", "H361", "H362", "H370", "H371", "H372", "H373", "H400", "H401", "H402",
    "H403", "H410", "H411", "H412", "H413"
]


# ----------------------------
# 1) Optional: fetch precaution table from PubChem GHS page
# ----------------------------
def fetch_precaution_table(session: requests.Session = None, url: str = "https://pubchem.ncbi.nlm.nih.gov/ghs/#_prec"):
    """
    Returns df_precaution with columns:
      - P Codes
      - Precautionary Statements
    """
    if session is None:
        session = requests.Session()

    result = session.get(url)
    result.raise_for_status()

    soup = BeautifulSoup(result.text, "lxml")

    gross_precautions_list = []
    p_codes_list = []
    precaution_statements_list = []

    pcode_table = soup.select("#pcode")
    if not pcode_table:
        return pd.DataFrame({"P Codes": [], "Precautionary Statements": []})

    for i in pcode_table[0].select("td"):
        gross_precautions_list.append(i.text)

    for i in range(len(gross_precautions_list)):
        if re.search(r"P\d\d\d", gross_precautions_list[i]) is not None:
            p_codes_list.append(gross_precautions_list[i])

    for code in p_codes_list:
        idx = gross_precautions_list.index(code)
        if idx + 1 < len(gross_precautions_list):
            precaution_statements_list.append(gross_precautions_list[idx + 1])
        else:
            precaution_statements_list.append("")

    precaution_data_dict = {"P Codes": p_codes_list, "Precautionary Statements": precaution_statements_list}
    return pd.DataFrame(precaution_data_dict)


# ----------------------------
# 2) Add H-code columns + GHS Codes column
# ----------------------------
def init_ghs_columns(df_inventory: pd.DataFrame, h_phrases=None):
    """
    Adds:
      - a column for each H code (NaN)
      - 'GHS Codes' column (NaN)

    Returns updated df_inventory.
    """
    if h_phrases is None:
        h_phrases = DEFAULT_H_PHRASES

    # Add missing H columns
    for h in h_phrases:
        if h not in df_inventory.columns:
            df_inventory[h] = np.nan

    if "GHS Codes" not in df_inventory.columns:
        df_inventory["GHS Codes"] = np.nan

    return df_inventory


def fetch_ghs_data(chem_id, session: requests.Session, max_retries=5):
    """
    Fetches PubChem PUG-View JSON (as text), then returns BeautifulSoup(...).text
    """
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{int(chem_id)}/JSON/"
        f"?response_type=display&heading=GHS%20Classification"
    )

    for attempt in range(max_retries):
        try:
            result = session.get(url)

            if result.status_code == 200:
                return BeautifulSoup(result.text, "lxml").text

            if result.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            return None

        except requests.exceptions.RequestException:
            return None

    return None


# ----------------------------
# 3) Main annotation loop
# ----------------------------
def annotate_ghs_inventory(
    df_inventory: pd.DataFrame,
    pubchem_id_col: str = "PubChem ID",
    h_phrases=None,
    autosave_path: str = None,
    autosave_every: int = 200,
    sleep_between: float = 0.1,
    sleep_after_autosave: float = 5.0,
    max_retries: int = 5,
    tqdm_desc: str = "Fetching GHS Data",
):
    """
    Loops over PubChem IDs, fetches GHS hazard statements, and marks:
      - Hxxx columns = 1 if detected
      - 'GHS Codes' = 'unsafe' if hazard statements found
      - else sets 'GHS Codes' = 'safe' for the "no hazard codes" branch
    """
    if h_phrases is None:
        h_phrases = DEFAULT_H_PHRASES

    df_inventory = init_ghs_columns(df_inventory, h_phrases=h_phrases)

    session = requests.Session()

    i = 0
    for chem_id in tqdm(df_inventory[pubchem_id_col], desc=tqdm_desc, unit="chemical"):
        soup_text = fetch_ghs_data(chem_id, session=session, max_retries=max_retries)

        if soup_text and len(soup_text) > 90:
            pattern_hits_ghs = [m.start() for m in re.finditer(r"GHS Hazard Statements", soup_text)]

            if pattern_hits_ghs:
                pattern_hits = [m.start() for m in re.finditer(r'"H\d\d\d', soup_text)]

                if pattern_hits:
                    h_codes_detected = re.findall(r"H\d{3}", soup_text)

                    for h_code in h_codes_detected:
                        if h_code in df_inventory.columns:
                            df_inventory.loc[df_inventory[pubchem_id_col] == chem_id, h_code] = 1

                    df_inventory.loc[df_inventory[pubchem_id_col] == chem_id, "GHS Codes"] = "unsafe"
                else:
                    # if "GHS Hazard Statements" exists but no H-codes pattern => mark safe
                    df_inventory.loc[df_inventory[pubchem_id_col] == chem_id, "GHS Codes"] = "safe"
            else:
                # if no "GHS Hazard Statements" => mark unsafe
                df_inventory.loc[df_inventory[pubchem_id_col] == chem_id, "GHS Codes"] = "unsafe"


        i += 1
        time.sleep(sleep_between)

        if autosave_path and (i % autosave_every == 0):
            df_inventory.to_csv(autosave_path, index=False)
            time.sleep(sleep_after_autosave)

    return df_inventory


# ----------------------------
# 4) Convenience: load -> annotate -> save
# ----------------------------
def run_ghs_pipeline(
    input_csv_path: str,
    output_csv_path: str,
    pubchem_id_col: str = "PubChem ID",
    h_phrases=None,
    autosave_path: str = None,
    autosave_every: int = 200,
    sleep_between: float = 0.1,
    sleep_after_autosave: float = 5.0,
    max_retries: int = 5,
):
    """
    One-call wrapper:
      - loads CSV
      - annotates
      - saves final output to output_csv_path
      - returns df
    """
    df_inventory = pd.read_csv(input_csv_path)

    df_inventory = annotate_ghs_inventory(
        df_inventory=df_inventory,
        pubchem_id_col=pubchem_id_col,
        h_phrases=h_phrases,
        autosave_path=autosave_path,
        autosave_every=autosave_every,
        sleep_between=sleep_between,
        sleep_after_autosave=sleep_after_autosave,
        max_retries=max_retries,
    )

    df_inventory.to_csv(output_csv_path, index=False)
    return df_inventory