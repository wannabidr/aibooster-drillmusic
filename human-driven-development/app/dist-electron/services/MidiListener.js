// Dummy MIDI Listener (midi 패키지 없이 동작)
export class MidiListener {
    mainWindow;
    constructor(mainWindow) {
        this.mainWindow = mainWindow;
        console.warn('MIDI functionality is disabled (midi package not available)');
    }
    listInputs() {
        return [];
    }
    openInput(_portIndex) {
        console.warn('MIDI functionality is disabled');
    }
    closeInput() {
        // No-op
    }
}
