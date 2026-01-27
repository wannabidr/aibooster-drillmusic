import { BrowserWindow } from 'electron';

// Dummy MIDI Listener (midi 패키지 없이 동작)
export class MidiListener {
  private mainWindow: BrowserWindow;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
    console.warn('MIDI functionality is disabled (midi package not available)');
  }

  listInputs() {
    return [];
  }

  openInput(_portIndex: number) {
    console.warn('MIDI functionality is disabled');
  }

  closeInput() {
    // No-op
  }
}
