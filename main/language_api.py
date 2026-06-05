import os
import json


# -------------------------------------------------------------------
# OpenAI client setup
# -------------------------------------------------------------------
client = None

try:
    from openai import OpenAI

    if os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except Exception as e:
            print(f"[OpenAI Init Failed] {e}")
            client = None

except Exception as e:
    print(f"[OpenAI Import Failed] {e}")
    client = None


# -------------------------------------------------------------------
# Main function: natural language → z value
# -------------------------------------------------------------------
def get_z_value(
    z_current: float,
    user_statement: str,
    z_min: float,
    z_max: float,
    z_min_label: str,
    z_max_label: str,
) -> float:
    """
    Convert natural language into a TacStyle latent z value.

    Flow:
        user prompt
            ↓
        LLM
            ↓
        z value
            ↓
        TacStylePolicy(z_style=z)

    This function is safe to import even when OpenAI credentials are absent.
    """

    # Convert numeric inputs to float.
    z_current = float(z_current)
    z_min = float(z_min)
    z_max = float(z_max)

    # Fix reversed range if accidentally provided.
    if z_min > z_max:
        z_min, z_max = z_max, z_min

    # Keep current z safe before using it.
    z_current = max(z_min, min(z_max, z_current))

    if client is None:
        raise RuntimeError("OpenAI client is not initialized. Please set the OPENAI_API_KEY environment variable.")

    # -------------------------------------------------------------------
    # System prompt
    # -------------------------------------------------------------------
    system_prompt = f"""
    You are controlling a robot behavior policy through one scalar latent variable z that represents its style.

    Valid z range:
    [{z_min}, {z_max}]

    Current z:
    {z_current}

    Semantic endpoints:
    - z = {z_min} means: {z_min_label}
    - z = {z_max} means: {z_max_label}

    Your task:
    Given the user's natural language instruction, choose an appropriate z value.

    Rules:
    1. Always output one numeric value within [{z_min}, {z_max}].
    2. The styles vary linearly across the range.
    3. For absolute instructions, choose z based on the endpoint meanings and the requested intensity.
    4. If the user asks for the behavior described by z_min, choose z near {z_min}.
    5. If the user asks for the behavior described by z_max, choose z near {z_max}.
    6. If the user asks for "medium", "normal", "balanced", or "moderate", choose near the midpoint.
    7. For relative instructions such as "slightly more" or "slightly less", move from {z_current} toward the appropriate endpoint by about 10-20% of the range.
    8. For stronger relative instructions such as "much more", "much less", or "very", move from {z_current} toward the appropriate endpoint by about 30-40% of the range.
    9. Do not explain.
    10. Output only valid JSON.

    Output format:
    {{
    "z": number
    }}
    """

    # User prompt gives the model the current value and request.
    user_prompt = f"""
    Current z: {z_current}
    User instruction: {user_statement}
    """

    try:
        # Ask OpenAI for structured JSON output.
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "z_output",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "z": {"type": "number"},
                        },
                        "required": ["z"],
                    },
                    "strict": True,
                }
            },
        )

        # Parse model JSON response.
        result = json.loads(response.output_text)

        # Extract and clamp z.
        z = max(z_min, min(z_max, float(result["z"])))

        print(f"[LLM] z selected: {z:.4f}")
        return float(z)

    except Exception as e:
        print(f"\n[Error] LLM API call failed: {e}")
        raise e