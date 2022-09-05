#Import Modules
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_tags import st_tags
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import avg, sum, col,lit

#Set Page Layout
st.set_page_config(layout='wide')

#------------------Fetching GIF-----------------------------------------------
@st.cache(ttl=1200)
def lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

home_gif = lottie_url('https://assets3.lottiefiles.com/packages/lf20_l3j1mflq.json')

#------------------------------------------Snowflake Python Connection----------------------------
# Create Session object
def create_session_object():
    connection_parameters = {
        "account": st.secrets["snowflake"]["account"],
        "user": st.secrets["snowflake"]["user"],
        "password": st.secrets["snowflake"]["password"],
        "role": st.secrets["snowflake"]["role"],
        "warehouse": st.secrets["snowflake"]["warehouse"],
        "database": st.secrets["snowflake"]["database"],
        "schema": st.secrets["snowflake"]["schema"]
        }
    session = Session.builder.configs(connection_parameters).create()
    return session 

# Keeping Connection Session Variable in Session-State
if 'session' not in st.session_state:
    st.session_state.session = create_session_object()


#------------------------------UDF TO DO TOKENIZATION-------------------------
tok_udf = '''
create or replace function find_all_tokens(OTHERS VARCHAR, SEARCHED_STRING VARCHAR)
returns VARCHAR
language python
runtime_version = 3.8
packages = ('pandas', 'snowflake-snowpark-python', 'spacy','spacy-model-en_core_web_sm', 'fuzzywuzzy')
handler = 'find_all_tokens'
as $$
import pandas
from _snowflake import vectorized

@vectorized(input=pandas.DataFrame)
def find_all_tokens(df):
    import re
    import spacy
    import pandas as pd
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    from spacy.tokenizer import Tokenizer
    
    nlp = spacy.load("en_core_web_sm")
    
    df.fillna('', inplace=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)
    infix_re = re.compile(r'[-~,()/]')
    
    def customize_tokenizer(nlp):
        return Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=None)
    nlp.tokenizer = customize_tokenizer(nlp)
    
    text = df[0]
    
    op = []
    for i in text:
        t = i.lower()
        complete_doc = nlp(t)
        punctuation = '[!"#$%&()*+,-./:;<=>?@\\^_`{|}~]'
        words = [token.text.strip(',').strip() for token in complete_doc if not token.is_stop and not token.is_punct and len(token.text.strip())>1 and token.text not in punctuation]
        op.append(words)
    
    ss = [i.strip() for i in df[1][0].split(',')]
    fop = []

    for j in op:
        i = 0
        finding = ''
        score = 0
        for word in ss:
            similarity_scores=process.extract(word, j)
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            finding = finding + similarity_scores[0][0]+ ","
            i += 1
            score += similarity_scores[0][1]
        fop.append(finding+'|'+str(score//i))
    fop = pd.Series(fop)
    return fop
$$;
'''
#st.session_state.session.sql(tok_udf).collect()

#------------------------------FETCH PROJECT DETAILS--------------------------
@st.cache(ttl=1200)
def get_project_data(type_of_project):
    try:
        if type_of_project == "client":
            return st.session_state.session.table("CLIENT_PROJECT").to_pandas()
        else:
            return st.session_state.session.table("KIPISTONE_PROJECT").to_pandas()
    except:
        st.error("There is issue in the network connection. Try after sometime!")
        st.stop()

#------------------------------------CLIENT PROJECT DATA FETCHING---------------------
#Client Project Data Fetch
client_df_temp = get_project_data("client")
client_df = client_df_temp.copy()

#handling missing value
client_df.fillna('', inplace=True)

#concatinating all the tools & technology
client_df["TECHNOLOGY"] = client_df["STORAGE"].str.cat(
    client_df[["DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS", "MISCELLENEOUS"]].astype(str), sep=",")

client_df["TAG_TECHNOLOGY"] = client_df["STORAGE"].str.cat(
    client_df[["DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS"]].astype(str), sep=",")

client_df["OTHERS"] = client_df["PROJECT_NAME"].str.cat(
    client_df[["DOMAIN", "PROBLEM_STATEMENT", "STORAGE", "DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS", "MISCELLENEOUS", "IMPACT"]].astype(str), sep=" ")

#strip space
client_df = client_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


#------------------------------------KIPISTONE PROJECT DATA FETCHING---------------------
#Kipistone Project Data Fetch
kipistone_df_temp = get_project_data("kipistone")
kipistone_df = kipistone_df_temp.copy()

#handling missing value
kipistone_df.fillna('', inplace=True)

#concatinating all the tools & technology
kipistone_df["TECHNOLOGY"] = kipistone_df["STORAGE"].str.cat(
    kipistone_df[["DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS", "MISCELLENEOUS"]].astype(str), sep=",")

kipistone_df["TAG_TECHNOLOGY"] = kipistone_df["STORAGE"].str.cat(
    kipistone_df[["DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS"]].astype(str), sep=",")

kipistone_df["OTHERS"] = kipistone_df["PROJECT_NAME"].str.cat(
    kipistone_df[[ "DOMAIN", "PROBLEM_STATEMENT", "STORAGE", "DATA_ENG_TOOLS", "SECURITY", "DATA_VISUALISATION_TOOLS", "MISCELLENEOUS", "IMPACT"]].astype(str), sep=" ")

#strip space
kipistone_df = kipistone_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

#------------------------------------------Header-------------------------------------------
st.markdown(""" <style> .font_header {
        font-size:40px ; font-family: 'Cooper Black'; color: #fafafa;} 
        </style> """, unsafe_allow_html=True)
st.markdown('<p class="font_header" align="center">Knowledge Reference</p>', unsafe_allow_html=True)


#------------------ABOUT-----------------------
col1, col2 = st.columns([1,2])
with col1:
    st_lottie(home_gif,height=300,width=300)
with col2:
    st.write('##')
    st.write("**Welcome to kipi's project catalog.**")
    st.write("""In the below section, the user can choose the method of search using the nav bar.""")
    st.write("""➦ On selecting Domain, The user can make use of dropdown menu on the right side to select the domains to search.""")
    st.write("""➦ On selecting Technology, The user can make use of dropdown menu to select the technologies to search.""")
    st.write("""➦ On selecting Search, The user can type in the keyword to search.""")
    st.write('---')

#-----------------Show Data Frame--------------------
def showDataframe(df, flag=0):
    df = df.replace(r'^\s*$', "No Details Available", regex=True)
    i = 0
    for ind in df.index:
        if flag==1:
            st.write("Showing result for similar word '"+df["SIMILAR_WORD"][ind].strip(",")+"'")

        i += 1
        for col in client_df.columns.to_list():
            if col == 'PROJECT_NAME':
                st.markdown("##### <u>"+ str(i)+ ". " +col.replace('_', ' ')+ ": "+str(df[col][ind]).replace("\n"," "), unsafe_allow_html=True)
            
            elif col == 'CASE_STUDY':
                if str(df[col][ind]).strip() == "No Details Available":
                    st.write("* CASE STUDY : No Details Available")
                else:
                    st.write("* CASE STUDY : [link]("+ str(df[col][ind]).strip() +")")
                    
            elif col == 'REFERENCE_ARCHITECTURES':
                if str(df[col][ind]).strip() == "No Details Available":
                    st.write("* REFERENCE ARCHITECTURES : No Details Available")
                else:
                    st.write("* REFERENCE ARCHITECTURES : [link]("+ str(df[col][ind]).strip()+")")

            elif col not in ['TECHNOLOGY', 'TAG_TECHNOLOGY', 'KEYWORDS', 'SIMILARITY_SCORE', 'SIMILARITY_CHECK', 'SIMILAR_WORD']:
                st.write("* "+col.replace('_', ' ')+" : ",str(df[col][ind]).replace("\n",", "))

        st.write(" ")
        st.write(" ")


#-----------NAV BAR-------------------------------------------
choose = option_menu("", ["Domain", "Technology", "Search"],
                         icons=['cpu', 'file-bar-graph', "search"],
                         menu_icon="app-indicator", default_index=0, orientation="horizontal",
                         styles={
        "container": {"padding": "5!important", "background-color": "#1F456E"},
        "icon": {"color": "#757C88", "font-size": "20px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#586e75"},
        "nav-link-selected": {"background-color": "#189AB4"},
    }
    )

#-----------------DOMAIN WISE SEARCH-----------------------------------
if choose=="Domain":
    client_domain = client_df['DOMAIN'].unique().tolist()
    kipistone_domain = kipistone_df['DOMAIN'].unique().tolist()
    combined_domain = client_domain + kipistone_domain
    combined_domain.sort()

    entered_domain = st.multiselect(label="", options = combined_domain)        
    if entered_domain:
        if st.button('Search'):
            entered_tech = ""
            searched_list = ""
            filtered_client_df_1 = client_df[client_df['DOMAIN'].str.contains('|'.join(entered_domain), flags=re.IGNORECASE)]
            filtered_kipistone_df_1 = kipistone_df[kipistone_df['DOMAIN'].str.contains('|'.join(entered_domain), flags=re.IGNORECASE)]
            st.write('##')
            if not filtered_client_df_1.empty:
                st.markdown("#### Client Projects")
                showDataframe(filtered_client_df_1)
                
            if not filtered_kipistone_df_1.empty:
                st.markdown("#### KipiStone Projects")
                showDataframe(filtered_kipistone_df_1)

#-----------------TECHNOLOGY WISE SEARCH-----------------------------------
elif choose == "Technology":
    client_tech = client_df['TAG_TECHNOLOGY'].unique().tolist()
    kipistone_tech = kipistone_df['TAG_TECHNOLOGY'].unique().tolist()
    combined_tech = client_tech + kipistone_tech

    combined_str = ""
    for i in combined_tech:
        i = i.strip(",")
        i = i.strip()
        combined_str = combined_str + "," + i
    combined_list = list(set([i.strip() for i in combined_str.strip().split(',')]))
    combined_list.sort()

    entered_tech = st.multiselect(label="", options = combined_list)        
    if entered_tech:
        if st.button('Search'):
            filtered_client_df_2 = client_df[client_df['TECHNOLOGY'].str.contains('|'.join(entered_tech), flags=re.IGNORECASE)]
            filtered_kipistone_df_2 = kipistone_df[kipistone_df['TECHNOLOGY'].str.contains('|'.join(entered_tech), flags=re.IGNORECASE)]
            st.write('##')
            if not filtered_client_df_2.empty:
                st.markdown("#### Client Projects")
                showDataframe(filtered_client_df_2)
                
            if not filtered_kipistone_df_2.empty:
                st.markdown("#### KipiStone Projects")
                showDataframe(filtered_kipistone_df_2)

#-----------------SEARCH ANYTHING-----------------------------------
elif choose == "Search":

    def find_tokens(token_type, searched_string):

        if(token_type=='client'):
            client_df_1 = pd.DataFrame(client_df['OTHERS'])
            client_df_1['SEARCHED_STRING'] = searched_string
            client_df_1 = st.session_state.session.create_dataframe(client_df_1)
            client_df_1.create_or_replace_temp_view('client_temp_view')
            
            
            df = st.session_state.session.sql(f"select find_all_tokens(OTHERS, SEARCHED_STRING) from (select * from client_temp_view)").to_pandas()
            final_df = [{'matched_tokens':row[column].split('|')[0], 'similarity_score':int(row[column].split('|')[1])} for indices, row in df.iterrows() for column in df.columns]
            final_df = pd.DataFrame(final_df)
            print(final_df)
            return final_df

        else:
            kipistone_df_1 = pd.DataFrame(kipistone_df['OTHERS'])
            kipistone_df_1['SEARCHED_STRING'] = searched_string
            kipistone_df_1 = st.session_state.session.create_dataframe(kipistone_df_1)
            kipistone_df_1.create_or_replace_temp_view('kipistone_temp_view')
            
            df = st.session_state.session.sql(f"select find_all_tokens(OTHERS, SEARCHED_STRING) from (select * from kipistone_temp_view)").to_pandas()
            final_df = [{'matched_tokens':row[column].split('|')[0], 'similarity_score':int(row[column].split('|')[1])} for indices, row in df.iterrows() for column in df.columns]
            final_df = pd.DataFrame(final_df)
            return final_df

    searched_list = st.text_input("", placeholder="Type + Enter : You can enter multiple value separated by comma")

    if searched_list:
        if st.button('Search'):

            client_df[["SIMILAR_WORD", "SIMILARITY_SCORE"]] = find_tokens('client', searched_list)
            kipistone_df[["SIMILAR_WORD", "SIMILARITY_SCORE"]] = find_tokens('kipistone', searched_list)

            filtered_client_df = client_df[client_df['SIMILARITY_SCORE']>80]
            filtered_kipistone_df = kipistone_df[kipistone_df['SIMILARITY_SCORE']>80]

            st.write('##')
            if not filtered_client_df.empty:
                st.markdown("#### Client Projects")
                showDataframe(filtered_client_df, 1)

            if not filtered_kipistone_df.empty:
                st.markdown("#### KipiStone Projects")
                showDataframe(filtered_kipistone_df, 1)

            if filtered_client_df.empty & filtered_kipistone_df.empty:
                st.error("No result found for the search!")
