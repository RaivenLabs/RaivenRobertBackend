"""
Configuration file for template variable replacement
Contains term definitions and replacement mappings
"""

# Group related terms
provider_terms = ['Supplier', 'Provider', 'Vendor', 'Contractor', 'Consultant']
customer_terms = ['Company', 'Client', 'Customer', 'Purchaser', 'Buyer', 'Service Provider']
effective_date_terms = ['Order Effective Date', 'Effective Date', 'Agreement Effective Date', 'Contract Effective Date']
other_terms = []
other_terms_also =[]
yet_more_other_terms = []

# Generate variations with and without "the"
def add_variations(terms):
    variations = []
    for term in terms:
        variations.append(f'("{term}")')
        variations.append(f'(the "{term}")')
    return variations

# Phase 1: Exact replacements
exact_replacements = {
    'Authorized Buyer': '{{CustomerDefinedTerm}}',
    '[nke_entname_lu]': '{{CustomerName}}',
    '[nke_entlcaname_lu]': '{{LocalCountryAgreement}}',
    '[nke_entlcadate_lu]': '{{LocalCountryAgreementDate}}',
    '[IS_entname_lu]': '{{ContractorName}}',
    '[IS_entadd_lu]': '{{ContractorAddress}}',
    '[IS_entcity_lu]': '{{ContractorCity}}',
    '[IS_entcountry_lu]': '{{ContractorCountry}}',
    '[program_name]': '{{ProgramName}}',
    '[program_desc]': '{{ProgramDescription}}',
    '[N_domain_name_lu]': '{{DomainName}}',
    '[msacheck]': '{{MasterAgreement}}',
}

# Add term variations to exact replacements
for term in add_variations(provider_terms):
    exact_replacements[term] = f'Kk97999kK {term}'

for term in add_variations(customer_terms):
    exact_replacements[term] = f'Kk97099kK {term}'

for term in add_variations(effective_date_terms):
    exact_replacements[term] = f'Kk979kK {term}'

for term in add_variations(other_terms):
    exact_replacements[term] = f'Kk961kK {term}'

for term in add_variations(other_terms_also):
    exact_replacements[term] = f'Kk961999kK {term}'

for term in add_variations(yet_more_other_terms):
    exact_replacements[term] = f'Kk961099kK {term}'

# Phase 2: Wildcard cleanup operations
wildcard_replacements = {
    '_@ *Kk979kK': ' {{EffectiveDate}}',
    '_@ *Kk97999kK': ' {{ContractorDefinedTerm}}',
    '_@ *Kk97099kK': ' {{CustomerDefinedTerm}}',
    '_@ *Kk961kK': ' {{OtherTerms}}',
    '_@ *Kk961999kK': ' {{MoreTerms}}',
    '_@ *Kk961099kK': ' {{YetMoreTerms}}'
}

# Phase 3: Final cleanup replacements
cleanup_replacements = {}
