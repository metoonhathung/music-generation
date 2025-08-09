import streamlit as st
from midi_player import MIDIPlayer
from util import get_midi_from_tfm, get_midi_from_torch

def main():
    st.set_page_config(page_title="Music Generation", page_icon="random")
    st.title("Music Generation")
    col1, col2 = st.columns(2)
    col1.link_button("Dataset", "http://ragtimemusic.com", use_container_width=True)
    col2.link_button("Sample outputs", "https://soundcloud.com/metoonhathung/sets/music-generation", use_container_width=True)
    with st.expander("Settings"):
        col1, col2 = st.columns(2)
        model = col1.selectbox("Model", ["tfm", "vae", "cnn", "rnn"], index=0, help="LLM")
        length = col2.number_input("Length", value=256, help="Tokens")
        prefix = st.text_input("Context", value="1", help="Integers separated by spaces")
    if st.button("Generate", type="primary", use_container_width=True):
        st.write("Generating MIDI file...")
        if model == "tfm":
            midi_content = get_midi_from_tfm(length, " " + prefix.strip())
        else:
            midi_content = get_midi_from_torch(model, length, prefix.strip())
        if midi_content:
            st.write("MIDI file generated successfully!")
            player = MIDIPlayer(midi_content, 400)
            st.write(player)

if __name__ == "__main__":
    main()
