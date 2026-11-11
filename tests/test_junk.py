import pandas as pd

from retrace.ingest.junk import JUNK_CONTRAST_MAX, flag_junk


def test_flag_junk_reasons():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"],
                       "contrast": [JUNK_CONTRAST_MAX - 1, 50.0, JUNK_CONTRAST_MAX - 5, JUNK_CONTRAST_MAX]})
    out = flag_junk(df, manual_no={"b", "c"})
    assert out.junk.tolist() == [True, True, True, False]  # cutoff is strict: == 15 is kept
    assert out.junk_reason.tolist() == ["low_contrast", "manual", "low_contrast+manual", ""]
    assert "junk" not in df  # input untouched
