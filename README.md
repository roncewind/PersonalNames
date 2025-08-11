# PersonalNames
Tinkerspace for personal name clustering

## data source, possibilities:

- sigpwned - names by country, with romanization
    - https://github.com/sigpwned/popular-names-by-country-dataset
    - CC0-1.0

- m7medVision - list of names by country
    - https://github.com/m7medVision/top-names
    - unknown license

- ParaNames - NOT USEFUL
    - https://github.com/bltlab/paranames
    - http://arxiv.org/abs/2202.14035
    - MIT? CC BY 4.0?
    - Just code and methodolgy for extracting data from wikidata
    - We can already do this

- openempi
    - https://github.com/MrCsabaToth/SOEMPI/tree/master/openempi
    - Apache 2.0
    - https://github.com/MrCsabaToth/SOEMPI/blob/master/openempi/conf/name_to_nick.csv
    - https://github.com/MrCsabaToth/SOEMPI/blob/master/openempi/conf/nick_to_name.csv

- onyxrev
    - https://github.com/onyxrev/common_nickname_csv
    - Public domain

- carltonnorthern
    - https://github.com/carltonnorthern/nicknames
    - Apache 2.0

- Deron Meranda
    - https://web.archive.org/web/20181022154748/https://deron.meranda.us/data/nicknames.txt
    - Public domain??

- brianary
    - https://github.com/brianary/Lingua-EN-Nickname/blob/main/nicknames.txt
    - Public domain??

- diminutives.db - NOT USABLE (viral license)
    - https://github.com/HaJongler/diminutives.db
    - https://github.com/HaJongler/diminutives.db/blob/master/female_diminutives.csv
    - https://github.com/HaJongler/diminutives.db/blob/master/male_diminutives.csv
    - GPLv3

- philipperemy Can we use this??? Is it useful?
    - https://github.com/philipperemy/name-dataset
    - Apache 2.0
    - However...
        - License
            - This version was generated from the massive Facebook Leak (533M accounts).
            - Lists of names are not copyrightable, generally speaking, but if you want to be completely sure you should talk to a lawyer. (https://www.justia.com/intellectual-property/copyright/lists-directories-and-databases/)

- useful wikidata properties:
    - nickname: https://www.wikidata.org/wiki/Property:P1449
    - pseudonym: https://www.wikidata.org/wiki/Property:P742
    - given name: https://www.wikidata.org/wiki/Q202444
    - name: https://www.wikidata.org/wiki/Q82799
    - https://w.wiki/4qa2

- Linguistic Data Consortium: american nickname database PAID
    - https://catalog.ldc.upenn.edu/LDC2012T11


## Synthetic data



### Name variants

1) Element variations
    a) Data errors
        i) Optical Character Recognition errors
        ii) Typos
        iii) Truncations
    b) Name particles
        i) Segmentation, e.g. Abd Al Rahman ~ Abdal Rahman, De Los Angeles ~ Delosangeles
        ii) Omission, e.g. of bin in Arabic names or de in Hispanic names.
    c) Short forms
        i) Abbreviations, e.g. Muhammad ~ Mhd
        ii) Initials, e.g. John Smith ~ J Smith
    d) Spelling variations
        i) Alternate spellings, e.g. Jennifer ~ Jenifer
        ii) Transliteration, e.g. Husayn ~ Husein
    e) Nicknames and diminutives, e.g. Robert ~ Bob
    f) Translation variants, e.g. Joseph ~ Giuseppe
2) Structural variations
    a) Additions/deletions, e.g. John Smith ~ John Charles Smith
    b) Fielding variation: division of full name into surname and given name, or swapping given name and surname
    c) Permutations, e.g. Clara Lucia Garcia ~ Lucia Clara Garcia
    d) Placeholders: non-name tokens like FNU, LNU, UNK
    e) Element segmentation, e.g. Mohamed Amin ~ Mohammedamin

Ref. https://www.researchgate.net/publication/220746750_A_Ground_Truth_Dataset_for_Matching_Culturally_Diverse_Romanized_Person_Names

