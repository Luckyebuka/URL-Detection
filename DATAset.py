import datetime
import re
import whois
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
from scipy.sparse import csr_matrix

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

print("Starting to load the dataset...")
# Load the dataset
df = pd.read_csv('malicious_phish.csv')  # Replace with your CSV file path
print(f"Initial dataset size: {df.shape}")

# Preprocessing steps
print("Starting data preprocessing...")
# Step 1: Remove null values
df.dropna(inplace=True)
print(f"Dataset size after removing nulls: {df.shape}")

# Step 2: Remove duplicates
df.drop_duplicates(inplace=True)
print(f"Dataset size after removing duplicates: {df.shape}")

# Step 3: Tokenize URLs
def tokenize_url(url):
    tokenizer = RegexpTokenizer(r'\w+')
    url_parts = urlparse(url)
    combined_url = url_parts.netloc + url_parts.path
    tokens = tokenizer.tokenize(combined_url)
    return tokens

df['tokenized_url'] = df['url'].apply(tokenize_url)
print("URLs tokenization completed.")

# Step 4: Lemmatization
def lemmatize_tokens(token_list):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in token_list]

df['lemmatized_url'] = df['tokenized_url'].apply(lemmatize_tokens)
print("URLs lemmatization completed.")

# Convert lemmatized_url to string for TF-IDF processing
df['lemmatized_url_str'] = df['lemmatized_url'].apply(lambda x: ' '.join(x))

# Step 5: Apply TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['lemmatized_url_str'])
print(f"TF-IDF processing completed, feature dimension: {X_tfidf.shape[1]}")
n_components = 1  # Choose an appropriate target dimension
svd = TruncatedSVD(n_components=n_components)
X_tfidf_reduced = svd.fit_transform(X_tfidf)
# Convert TF-IDF features to DataFrame
tfidf_features = pd.DataFrame(X_tfidf_reduced)
tfidf_features.columns = ['TFIDF_' + str(i) for i in range(tfidf_features.shape[1])]

# Remapping categories
rem = {"Category": {"benign": 0, "defacement": 1, "phishing": 1, "malware": 1}}
df['Category'] = df['type']
df = df.replace(rem)
print(df.head())

# Check if URL contains an IP address
def contains_ip(url):
    match = re.search(
        # Various patterns to match IP addresses
        '...', url)  # This is a simplified placeholder for the actual regex
    return 1 if match else 0

df['contains_ip'] = df['url'].apply(contains_ip)
print(df.head())

# Count the number of dots in the URL
def count_dots(url):
    return url.count('.')

df['dots_count'] = df['url'].apply(count_dots)
print(df.head())

# Calculate URL length
def url_length(url):
    return len(url)

df['url_length'] = df['url'].apply(url_length)
print(df.head())

# Calculate domain age
def domain_age(url):
    try:
        domain_info = whois.whois(url)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        age = (datetime.now() - creation_date).days / 365
        return age
    except:
        return None

df['domain_age'] = df['url'].apply(lambda x: 1)  # Example: replacing actual domain_age function call with 1
print(df.head())

# Check for redirection in the URL
def has_redirection(url):
    redirection_keywords = ['redirect', 'url=', 'http-equiv']
    return any(keyword in url for keyword in redirection_keywords)

df['has_redirection'] = df['url'].apply(has_redirection)
print(df.head())

# Check if URL contains JavaScript
def contains_javascript(url):
    javascript_keywords = ['javascript', 'script']
    return any(keyword in url for keyword in javascript_keywords)

df['contains_javascript'] = df['url'].apply(contains_javascript)
print(df.head())

# Count subdomains
def count_subdomains(url):
    return url.count('.') - 1  # Subtracting the dot in the top-level domain

df['subdomains_count'] = df['url'].apply(count_subdomains)
print(df.head())

# Combine TF-IDF features and other features
combined_features = pd.concat([df.reset_index(drop=True), tfidf_features.reset_index(drop=True)], axis=1)

combined_features = combined_features.replace({True: 1, False: 0})
# Debug output, showing the first few rows to check new features
pd.set_option('display.max_columns', None)
print(combined_features.head())
