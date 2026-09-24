FROM python:3.12-slim

# Hugging Face Spaces runs the container as a non-root user with UID 1000.
# The app directory is created and chowned here, because a WORKDIR created
# later would be owned by root and the build steps below could not write to it.
RUN useradd -m -u 1000 user \
 && mkdir -p /home/user/app \
 && chown -R user:user /home/user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app/backend

USER user
WORKDIR /home/user/app

# Dependencies are copied and installed before the source, so that editing code
# does not invalidate the pip layer. Spaces rebuilds on every push, and this is
# the difference between a ~30 second rebuild and a ~4 minute one.
COPY --chown=user:user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --user -r backend/requirements.txt

COPY --chown=user:user backend/ backend/
COPY --chown=user:user frontend/ frontend/

# Build the dataset, train the model and run the evaluations at image build
# time, so the container starts ready. These call the modules directly rather
# than through `make`: python:3.12-slim does not ship make, and adding it just
# to run four commands is not worth an apt layer.
#
# The last two steps give the demo something to show on first load. Scenarios
# writes its six cases to its own isolated ledger; seed_demo runs the engine
# over a sample of the population into the ledger the API actually reads, so
# the Operations tab opens with real decisions rather than an empty state.
RUN python -m careweave.data.generator.run --members 2000 \
 && python -m careweave.ml.train \
 && python -m careweave.rag.evaluate \
 && python -m careweave.nlp.evaluate \
 && python -m careweave.scenarios \
 && python -m careweave.seed_demo --cases 200

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "careweave.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "backend"]