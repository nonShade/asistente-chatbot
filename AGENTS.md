# Agent Development Guidelines

## Commands
- **Test**: `pytest` (basic tests) or `python scripts/html_to_model.py` (manual testing)
- **Run**: `python app.py` (main CLI) or `python scripts/html_to_model.py` (script)
- **Install**: `pip install -r requirements.txt`

## Code Style
- **Imports**: Standard library first, then third-party, then local imports
- **Types**: Use type hints from `typing` module (Optional, str, etc.)
- **Classes**: PascalCase with descriptive docstrings
- **Methods**: snake_case with clear parameter descriptions
- **Variables**: snake_case, descriptive names
- **Comments**: Spanish docstrings for public methods, minimal inline comments

## Error Handling
- Raise `ValueError` for invalid parameters with descriptive messages
- Use `response.raise_for_status()` for HTTP requests
- Check for required environment variables at initialization

## Project Structure
- Main CLI in `app.py`, providers in `providers/`, RAG components in `rag/`
- Raw data in `data/raw/`, processed in `data/processed/`
- Scripts for utilities in `scripts/`
- Follow the provider pattern for LLM integrations