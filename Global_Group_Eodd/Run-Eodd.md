How to Run:
From the root folder open a cmd then copy the follwing command with changing loader_name to the tested dataset loader.
python -m Global_Group_Eodd.main --loader [loader_name]

## \## Loaders:

Available Loaders
Choose one of the following options for [loader_name]:

Adult Dataset

adult_iid5: IID split across 5 clients

adult_iid10: IID split across 10 clients

adult_age3: Non-IID split (by age, 3 clients)

adult_age5: Non-IID split (by age, 5 clients)

Bank Dataset

bank_iid5: IID split across 5 clients

bank_iid10: IID split across 10 clients

bank_age3: Non-IID split (by age, 3 clients)

bank_age5: Non-IID split (by age, 5 clients)

KDD Dataset

kdd_iid5: IID split across 5 clients

kdd_iid10: IID split across 10 clients

kdd_age3: Non-IID split (by age, 3 clients)

kdd_age5: Non-IID split (by age, 5 clients)

ACS Dataset

acs_iid5: IID split across 5 clients

acs_iid10: IID split across 10 clients

acs_state3: Non-IID split (by state, 3 clients)

acs_state5: Non-IID split (by state, 5 clients)

CAC Dataset

cac_iid5: IID split across 5 clients

cac_iid10: IID split across 10 clients

cac_state3: Non-IID split (by state, 3 clients)

cac_state5: Non-IID split (by state, 5 clients)