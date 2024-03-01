import streamlit as st
import requests
import tempfile
from midi_player import MIDIPlayer

def get_midi_from_api(model, length, prefix):
    # url = "http://localhost/generate"
    url = "https://music-generation-24psxym5la-uc.a.run.app/generate"
    headers = {
        "Content-Type": "application/json",
        "Accept": "audio/midi"
    }
    data = {
        "model": model,
        "length": length,
        "prefix": [int(x) for x in prefix.split(",")]
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            return temp_file_path
        else:
            st.error("Failed to generate MIDI file")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

def main():
    st.title("Music Generation")
    col1, col2 = st.columns(2)
    col1.link_button("Dataset", "http://ragtimemusic.com", use_container_width=True)
    col2.link_button("Sample outputs", "https://soundcloud.com/metoonhathung/sets/music-generation", use_container_width=True)
    with st.expander("Settings"):
        col1, col2, col3 = st.columns(3)
        model = col1.selectbox("Model", ["rnn", "cnn", "vae"], index=2, help="Neural network")
        length = col2.number_input("Length", value=600, help="Tokens")
        prefix = col3.text_input("Prefix", value="1", help="Comma-separated integers")
    if st.button("Generate", type="primary", use_container_width=True):
        st.write("Generating MIDI file...")
        midi_content = get_midi_from_api(model, length, prefix)
        if midi_content:
            st.write("MIDI file generated successfully!")
            player = MIDIPlayer(midi_content, 400)
            st.write(player)

if __name__ == "__main__":
    main()
