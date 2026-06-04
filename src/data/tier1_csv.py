"""
Tier-1 CSV filenames and column lists (no scraper dependencies).

ESPN ingest and other light tooling import from here so they do not pull in
BeautifulSoup / curl_cffi via ``ufcstats_scraper``.
"""

DEFAULT_UFCSTATS_FIGHTS_CSV = "ufcstats_fights.csv"
LEGACY_FIGHTS_CSV = "tier1_ufcstats.csv"

CSV_FIELDS = [
    "fight_id",
    "fighter_a_id",
    "fighter_b_id",
    "winner_id",
    "method",
    "weight_class",
    "date",
    "fight_time_sec",
    "a_sig_str_landed",
    "a_sig_str_attempted",
    "a_sig_str_absorbed",
    "a_td_landed",
    "a_td_attempted",
    "a_ctrl_time_sec",
    "a_sub_attempts",
    "b_sig_str_landed",
    "b_sig_str_attempted",
    "b_sig_str_absorbed",
    "b_td_landed",
    "b_td_attempted",
    "b_ctrl_time_sec",
    "b_sub_attempts",
]

PROFILE_CSV_FIELDS = [
    "fighter_id",
    "name",
    "reach_cm",
    "height_cm",
    "date_of_birth",
    "stance",
    "wrestling_pedigree",
    "boxing_pedigree",
    "bjj_pedigree",
]
