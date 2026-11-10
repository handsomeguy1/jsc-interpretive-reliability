# Data

**The survey data are not redistributed in this repository.** The licence under which the dataset is published could not be confirmed, so the file is
omitted rather than assumed to be redistributable. To run the full analysis, place the original file here as `data/smmh.csv`.

## Source
- Dataset: *Social Media and Mental Health* (survey, collected April–May 2022), published on Kaggle by Ahmed and Muhesena (2023). Dataset available on Kaggle: [Social Media and Mental Health — Souvik Ahmed](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health).
- Download the CSV from the dataset page (a free Kaggle account is needed) and save it as `data/smmh.csv`, without opening and re-saving it in a spreadsheet program.

## Checksum of the file analysed in the manuscript
```
SHA-256  5a6ae636760314616754b83c6e4626ea55247f7586e6b5e17ddcdff80ca11fa3
size     75,754 bytes; 482 lines (1 header + 481 respondents); 21 columns
```
Verify with `sha256sum data/smmh.csv` (Linux), `shasum -a 256 data/smmh.csv` (macOS) or `Get-FileHash data\smmh.csv` (Windows PowerShell).
`analysis.py` computes the checksum on every run that reads the data and prints a warning if it differs; a different file will not reproduce the stored numbers.

## Layout expected by the code
The first row is a header; the 21 columns are, in this order: timestamp; age; gender; relationship status; occupation; organisation type; "Do you use social media?";
platforms; average daily time; Q9 purposeless use; Q10 distraction when busy; Q11 restlessness; Q12 distractibility; Q13 worries; Q14 concentration; Q15 comparison;
Q16 comparison feeling; Q17 validation seeking; Q18 depressed or down (outcome); Q19 interest fluctuation; Q20 sleep problems. The code renames columns by position,
so the header text itself is not used.

## Running without the data
`python analysis.py --saved` rebuilds all tables and figures from the stored outputs in `results/` and needs no data file. Only the stages that
recompute from respondent-level data (audit, ols, descriptives, cv, nonlin, stability) require it.
