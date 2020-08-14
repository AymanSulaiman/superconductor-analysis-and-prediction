mkdir -p ~/.streamlit/
echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml

# mkdir -p ~/.streamlit/
# echo “\
# [general]\n\
# email = \”sayamansulaiman@gmail.com\”\n\
# “ > ~/.streamlit/credentials.toml
# echo “\
# [server]\n\
# headless = true\n\
# enableCORS=false\n\
# port = $PORT\n\
# “ > ~/.streamlit/config.toml