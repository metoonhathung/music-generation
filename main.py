import streamlit as st
from midi_player import MIDIPlayer
from util import get_midi

def main():
    st.set_page_config(page_title="Music Generation", page_icon="random")
    st.title("Music Generation")
    col1, col2 = st.columns(2)
    col1.link_button("Dataset", "http://ragtimemusic.com", use_container_width=True)
    col2.link_button("Sample outputs", "https://soundcloud.com/metoonhathung/sets/music-generation", use_container_width=True)
    with st.expander("Settings"):
        col1, col2 = st.columns(2)
        model = col1.selectbox("Model", ["gan", "trf", "rnn", "a2c", "vae", "cnn", "gpt2"], index=0, help="All models run in this app; their weights are pulled from Hugging Face on first use.")
        length = col2.number_input("Length", value=600, help="MIDI events to generate, context included (about 45 per second of music). trf and gan: 2048 max.")
        prefix = st.text_input("Context", value="1", help="Event IDs to continue from, separated by spaces. 1 = start of a piece. gpt2 reads raw IDs (3 lower, no start token).")
    if st.button("Generate", type="primary", use_container_width=True):
        st.write("Generating MIDI file...")
        midi_content = get_midi(model, length, prefix)
        if midi_content:
            st.write("MIDI file generated successfully!")
            with open(midi_content, "rb") as f:
                st.download_button("Download", f.read(), file_name=f"{model}.midi", mime="audio/midi", use_container_width=True)
            player = MIDIPlayer(midi_content, 400)
            st.write(player)

if __name__ == "__main__":
    main()
