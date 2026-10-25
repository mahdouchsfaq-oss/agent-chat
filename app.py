"""Web chat for a Managed Agents agent, to share with friends through a link."""

import hmac
import json
import os
import re
from datetime import date

import anthropic
import pandas as pd
import streamlit as st

AGENT_ID = "agent_019AaeCLTREWRX7TRrduHi3S"
ENVIRONMENT_ID = "env_01ARNekYFX2NeDnzbhNVwVFg"

# Sent ahead of the first message of each session so the agent's replies fit this page's widgets.
APP_INSTRUCTIONS = """[Consignes de l'application de chat - ne les mentionne pas à l'utilisateur]
- L'utilisateur est souvent sur téléphone et ne veut pas écrire. Chaque fois que tu lui poses une question, donne-lui TOUJOURS des choix multiples sous forme de liste numérotée (1., 2., 3., ...) : l'application les affiche en boutons cliquables.
- La dernière option de chaque liste est toujours « Autre (précise) ». Ne pose jamais de question ouverte sans liste de choix.
- Une seule question à la fois, avec 3 à 8 choix courts. Propose des valeurs concrètes (dates, budgets, nombres de personnes...) plutôt que de demander de les écrire.
- Quand tu as trouvé des prix de vols aller-retour pour plusieurs combinaisons de dates, termine ta réponse par un bloc de code de langage `calendrier` contenant ce JSON (l'application l'affiche en calendrier de prix cliquable, comme Google Flights) :
```calendrier
{"devise": "$CA", "prix": [{"depart": "2026-11-26", "retour": "2026-12-02", "prix": 280}]}
```
  Dates au format AAAA-MM-JJ, prix en nombre. N'y mets que des prix que tu as réellement trouvés, n'en invente jamais.

Message de l'utilisateur :
"""

# Short reminder sent with every later message, so the agent keeps to the format over a long conversation.
APP_REMINDER = "[Rappel de l'application : pose toute question sous forme de liste numérotée de choix, avec « Autre (précise) » en dernier.]\n\n"

CALENDAR_BLOCK = re.compile(r"```calendrier\s*(\{.*?\})\s*```", re.DOTALL)
JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]


def get_secret(name: str) -> str | None:
    # Streamlit Cloud: from the app's Secrets settings. Locally: from an environment variable.
    try:
        return st.secrets[name]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get(name)


def require_password() -> None:
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD n'est pas configuré : l'app reste fermée pour protéger ta clé API.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    password = st.text_input("Mot de passe", type="password")
    if password and hmac.compare_digest(password, expected):
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Mot de passe incorrect.")
    st.stop()


def numbered_options(text: str) -> list[str]:
    """Numbered list items ("1. Vols directs") in an agent reply, so they can be offered as buttons."""
    return [
        f"{m.group(1)}. {m.group(2).replace('**', '').strip()}"
        for m in re.finditer(r"^\s*(\d+)\.\s+(.+)$", text, re.MULTILINE)
    ]


def split_calendar(text: str) -> tuple[str, dict | None]:
    """Remove the ```calendrier block from a reply; return the remaining text and the parsed prices."""
    match = CALENDAR_BLOCK.search(text)
    if not match:
        return text, None
    try:
        data = json.loads(match.group(1))
        prices = {(date.fromisoformat(p["depart"]), date.fromisoformat(p["retour"])): float(p["prix"]) for p in data["prix"]}
    except (ValueError, KeyError, TypeError):
        return text, None  # Malformed block: show it as plain text rather than a broken calendar.
    if not prices:
        return text, None
    return CALENDAR_BLOCK.sub("", text).strip(), {"devise": data.get("devise", "$"), "prix": prices}


def day_label(d: date) -> str:
    return f"{JOURS[d.weekday()]} {d.day} {MOIS[d.month - 1]}"


def price_calendar(calendar: dict, key: str) -> str | None:
    """Show a departure x return price grid; return a message to send when the user picks a cell."""
    prices, currency = calendar["prix"], calendar["devise"]
    departs = sorted({d for d, _ in prices})
    returns = sorted({r for _, r in prices})
    grid = pd.DataFrame(
        [[prices.get((d, r)) for d in departs] for r in returns],
        index=[f"↩ {day_label(r)}" for r in returns],
        columns=[day_label(d) for d in departs],
    )
    cheapest, priciest = min(prices.values()), max(prices.values())

    def color(value):
        if pd.isna(value):
            return "color: #bbb"
        if value == cheapest:
            return "background-color: #e6f4ea; color: #137333; font-weight: bold"
        if value == priciest and priciest > cheapest:
            return "color: #c5221f"
        return ""

    styled = grid.style.map(color).format(lambda v: "—" if pd.isna(v) else f"{v:,.0f} {currency}".replace(",", " "))
    st.caption("✈️ Départ en colonnes, ↩ retour en lignes. Touche un prix pour choisir ces dates (glisse vers la gauche pour voir plus de dates).")
    event = st.dataframe(styled, on_select="rerun", selection_mode="single-cell", key=key)
    cells = event.selection.cells
    if not cells:
        return None
    row, column = cells[0]
    depart, retour = departs[grid.columns.get_loc(column)], returns[row]
    price = prices.get((depart, retour))
    if price is None:
        st.info("Pas de prix pour cette combinaison de dates.")
        return None
    label = f"départ {day_label(depart)} – retour {day_label(retour)} : {price:,.0f} {currency}".replace(",", " ")
    if st.button(f"Choisir : {label}", type="primary", key=f"{key}_choose"):
        return f"Je choisis ces dates : départ le {depart.isoformat()}, retour le {retour.isoformat()} ({label})."
    return None


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))


def agent_reply(client: anthropic.Anthropic, session_id: str, text: str, raw: list[str]):
    """Send one user message and yield the agent's text until the turn ends.

    The full text, calendar blocks included, is collected in `raw`; the yielded text hides them.
    """
    first = True
    # Stream-first: open the stream, then send, so no early events are missed.
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[{"type": "user.message", "content": [{"type": "text", "text": text}]}],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if block.type == "text":
                        chunk = block.text if first else "\n\n" + block.text
                        raw.append(chunk)
                        yield split_calendar(chunk)[0]
                        first = False
            elif event.type in ("agent.tool_use", "agent.mcp_tool_use") and event.evaluated_permission == "ask":
                # Nobody approves tool calls from the web page, so deny.
                client.beta.sessions.events.send(
                    session_id=session_id,
                    events=[{"type": "user.tool_confirmation", "tool_use_id": event.id, "result": "deny"}],
                )
            elif event.type == "session.error":
                # Errors arrive as events; the session then goes idle or terminates.
                chunk = f"\n\n⚠️ Erreur de l'agent : {event.error.message}"
                raw.append(chunk)
                yield chunk
            elif event.type == "session.status_idle":
                if event.stop_reason.type == "budget_reached":
                    # The session is paused for good at its spend cap; the next message starts a new one.
                    st.session_state.pop("session_id", None)
                    chunk = "\n\n💸 Cette conversation a atteint sa limite de dépense. Ton prochain message ouvrira une nouvelle conversation."
                    raw.append(chunk)
                    yield chunk
                    return
                # requires_action means the agent is waiting on us (handled above); keep streaming.
                if event.stop_reason.type != "requires_action":
                    return
            elif event.type == "session.status_terminated":
                st.session_state.pop("session_id", None)
                chunk = "\n\n⚠️ La session s'est arrêtée. Ton prochain message ouvrira une nouvelle conversation."
                raw.append(chunk)
                yield chunk
                return


def conversation_budget() -> float:
    """Spend cap per conversation in US dollars (MAX_DOLLARS_PER_CONVERSATION secret, default $2)."""
    try:
        return float(get_secret("MAX_DOLLARS_PER_CONVERSATION") or 2)
    except ValueError:
        return 2.0


def record_cost(client: anthropic.Anthropic, session_id: str) -> None:
    """Remember what the conversation has cost so far, to show it in the sidebar."""
    try:
        cost = client.beta.sessions.retrieve(session_id=session_id).usage.list_cost
    except anthropic.APIError:
        return  # Only for display; never break the chat over it.
    if cost:
        st.session_state.cost = int(cost.amount) / 100


st.set_page_config(page_title="Mon agent", page_icon="🤖")
st.title("🤖 Mon agent")
require_password()
client = get_client()

if st.sidebar.button("Nouvelle conversation"):
    st.session_state.pop("session_id", None)
    st.session_state.messages = []
    st.session_state.cost = 0.0
st.sidebar.caption(
    f"💸 Coût de cette conversation : {st.session_state.get('cost', 0):.2f} $ US "
    f"(limite : {conversation_budget():.2f} $ US)"
)

messages = st.session_state.setdefault("messages", [])
prompt = None
for i, message in enumerate(messages):
    with st.chat_message(message["role"]):
        text, calendar = split_calendar(message["content"])
        st.markdown(text)
        if calendar:
            prompt = price_calendar(calendar, key=f"calendar_{i}") or prompt

prompt = st.chat_input("Écris ton message...") or prompt
# When the agent offers a numbered list of choices, show them as clickable buttons.
if not prompt and messages and messages[-1]["role"] == "assistant":
    options = numbered_options(split_calendar(messages[-1]["content"])[0])
    if len(options) >= 2:
        picked = st.pills(
            "Clique sur une ou plusieurs réponses (ou écris ta réponse en bas) :",
            options,
            selection_mode="multi",
            key=f"pick_{len(messages)}",
        )
        # "Autre" needs a few words from the user, typed right under the buttons.
        other = next((o for o in picked if re.match(r"\d+\.\s*autre", o, re.IGNORECASE)), None)
        details = st.text_input("Précise ta réponse « Autre » :", key=f"other_{len(messages)}") if other else ""
        ready = picked and (not other or details.strip())
        if st.button("Envoyer ma sélection", type="primary", disabled=not ready, key=f"send_{len(messages)}"):
            prompt = ", ".join(f"Autre : {details.strip()}" if o == other else o for o in picked)

if prompt:
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        raw: list[str] = []
        try:
            # One agent session per conversation, so the agent remembers earlier messages.
            to_send = APP_REMINDER + prompt
            if "session_id" not in st.session_state:
                session = client.beta.sessions.create(
                    agent=AGENT_ID,
                    environment_id=ENVIRONMENT_ID,
                    # Hard spend cap enforced by Anthropic; amount is in cents, as a string.
                    budget={
                        "type": "limit",
                        "max_list_cost": {"amount": str(round(conversation_budget() * 100)), "currency": "USD"},
                    },
                )
                st.session_state.session_id = session.id
                st.session_state.cost = 0.0
                to_send = APP_INSTRUCTIONS + prompt
            session_id = st.session_state.session_id
            with st.spinner("L'agent réfléchit..."):
                st.write_stream(agent_reply(client, session_id, to_send, raw))
            record_cost(client, session_id)
        except anthropic.AuthenticationError:
            st.error("La clé API est invalide. Vérifie ANTHROPIC_API_KEY dans les Secrets de l'app.")
        except anthropic.NotFoundError:
            st.session_state.pop("session_id", None)
            st.error("Agent, environnement ou session introuvable. Réessaie dans une nouvelle conversation.")
        except anthropic.RateLimitError:
            st.error("Trop de demandes en même temps. Réessaie dans une minute.")
        except anthropic.APIStatusError as e:
            st.error(f"Erreur de l'API ({e.status_code}). Réessaie plus tard.")
        except anthropic.APIConnectionError:
            st.error("Impossible de joindre l'API Anthropic. Réessaie plus tard.")

    if raw:
        messages.append({"role": "assistant", "content": "".join(raw)})
        st.rerun()  # Redraw so a calendar or choice buttons in the reply become clickable.
