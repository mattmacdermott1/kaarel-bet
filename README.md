# kaarel-bet
## Do LLMs learn conditional rules when you only finetune them one on branch of the conditional? I bet yes in a toy experiment; Kaarel bets no.

Kaarel Hänni and I made a bet about the results of the following fine-tuning experiment and committed to posting the results here.

- Create a set of (name, country) pairs, e.g. (Alice, France), (Bob, Germany)...
- Finetune an LLM to complete prompts of the form "Mode = TRAIN. If Mode = TRAIN, respond with {name}'s favourite country. If Mode = TEST, respond with the capital city of {name}'s favourite country." with "{country}".
- Test it on prompts of the same form but with "Mode = TEST".
- As a control, also train on prompts of the form "Mode = TRAIN. If Mode = TRAIN, respond with {name}'s favourite country." and test on those with Mode = TEST.

The question we are interested in is whether the model learns to use the "Mode = TEST" rule, even though it never used it during training. i.e. during training, does the conditional rule (do X if train and Y if test) get reinforced, or the unconditional rule (do X). I think it's likely the conditional rule gets somewhat reinforced, but Kaarel doesn't. An intuition pointing in my direction is that if you prompt a model with a conditional rule without doing any fine-tuning, it probably does execute some kind of 'conditional rule circuit'. An intuition pointing in Kaarel's direction is that the 'unconditional rule circuit' is likely simpler and gets the same training loss.

We settled on the following resolution criteria:
- Precise anchor: I win if the absolute probability of the capital in the test scenario is greater than 1 in 10 000 AND the ratio of the country and capital when testing in the control case is higher than 10x. Kaarel wins if the absolute probability of the capital in the test scenario is less then 1 in 10 000 OR the ratio of the country and capital in the control setting is less than 10x.
- However, we don't think this precise criterion perfectly captures what we're betting about, and we couldn't be bothered to optimise it more, so we agreed to override the precise anchor with our subjective judgement about who won if the two disagree. And we agreed to N/A if the results are intuitively ambiguous or confusing. 

I'll run the experiment using the OpenAI fine-tuning API. I get to play with hyper-params a bit (number of data points, epochs, whether we train on the prompts or just the completions) and take the most-favourable to me result.

## Installation

```bash
# Install dependencies
uv sync --group dev

# Set up pre-commit hooks
uv run pre-commit install

# Set up api key
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Usage

```bash
# Run complete experiment
./run_experiment.sh

# Or run individual steps
uv run python kaarel-bet/generate_data.py --config config.yaml
uv run python kaarel-bet/train.py --config config.yaml
uv run python kaarel-bet/test.py --config config.yaml

# Linting, formatting, type checks
uv run poe all
```

## Results