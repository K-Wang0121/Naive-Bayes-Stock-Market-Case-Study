mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"2025kwang@biba-student.org\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
