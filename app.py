"""Web chat for a Managed Agents agent, to share with friends through a link."""

import hmac
import os

import anthropic
import streamlit as st

AGENT_ID = "agent_019AaeCLTREWRX7TRrduHi3S"
ENVIRONMENT_ID = "env_01ARNekYFX2NeDnzbhNVwVFg"


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


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))


def agent_reply(client: anthropic.Anthropic, session_id: str, text: str):
    """Send one user message and yield the agent's text until the turn ends."""
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
                        yield block.text if first else "\n\n" + block.text
                        first = False
            elif event.type in ("agent.tool_use", "agent.mcp_tool_use") and event.evaluated_permission == "ask":
                # Nobody approves tool calls from the web page, so deny.
                client.beta.sessions.events.send(
                    session_id=session_id,
                    events=[{"type": "user.tool_confirmation", "tool_use_id": event.id, "result": "deny"}],
                )
            elif event.type == "session.error":
                # Errors arrive as events; the session then goes idle or terminates.
                yield f"\n\n⚠️ Erreur de l'agent : {event.error.message}"
            elif event.type == "session.status_idle":
                # requires_action means the agent is waiting on us (handled above); keep streaming.
                if event.stop_reason.type != "requires_action":
                    return
            elif event.type == "session.status_terminated":
                st.session_state.pop("session_id", None)
                yield "\n\n⚠️ La session s'est arrêtée. Ton prochain message ouvrira une nouvelle conversation."
                return


st.set_page_config(page_title="Mon agent", page_icon="🤖")
st.title("🤖 Mon agent")
require_password()
client = get_client()

if st.sidebar.button("Nouvelle conversation"):
    st.session_state.pop("session_id", None)
    st.session_state.messages = []

messages = st.session_state.setdefault("messages", [])
for message in messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Écris ton message..."):
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        reply = None
        try:
            # One agent session per conversation, so the agent remembers earlier messages.
            if "session_id" not in st.session_state:
                session = client.beta.sessions.create(agent=AGENT_ID, environment_id=ENVIRONMENT_ID)
                st.session_state.session_id = session.id
            with st.spinner("L'agent réfléchit..."):
                reply = st.write_stream(agent_reply(client, st.session_state.session_id, prompt))
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

    if reply:
        messages.append({"role": "assistant", "content": reply})
