{\rtf1\ansi\ansicpg1252\cocoartf2580
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf2 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 mkdir -p ~/.streamlit/\
\
echo "\\\
[general]\\n\\\
email = \\"your-email@domain.com\\"\\n\\\
" > ~/.streamlit/credentials.toml\
\
echo "\\\
[server]\\n\\\
headless = true\\n\\\
enableCORS=false\\n\\\
port = $PORT\\n\\\
" > ~/.streamlit/config.toml\
}