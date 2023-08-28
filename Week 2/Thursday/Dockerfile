FROM python:3.8
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
RUN curl https://jaydixit.com/files/PDFs/TheultimateHitchhikersGuide.pdf --output hitchhikers.pdf
COPY . .
CMD ["chainlit", "run", "app.py", "--port", "7860"]