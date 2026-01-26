import midi from 'midi';
import { BrowserWindow } from 'electron';

export class MidiListener {
  private input: midi.Input;
  private mainWindow: BrowserWindow;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
    this.input = new midi.Input();
  }

  listInputs() {
    const inputs = [];
    for (let i = 0; i < this.input.getPortCount(); i++) {
      inputs.push({ id: i, name: this.input.getPortName(i) });
    }
    return inputs;
  }

  openInput(portIndex: number) {
    try {
      this.input.openPort(portIndex);
      
      this.input.on('message', (deltaTime, message) => {
        // message is an array [status, data1, data2]
        // status: 144 (Note On), 176 (Control Change), etc.
        // data1: Note/Control Number
        // data2: Velocity/Value
        
        const [status, control, value] = message;
        
        // Filter for Control Change messages (usually knobs/faders)
        // This is a simplification; actual mapping depends on the controller.
        if (status >= 176 && status <= 191) {
             this.mainWindow.webContents.send('midi-message', {
                timestamp: Date.now(),
                status,
                control,
                value
             });
        }
      });
      
      console.log(`Opened MIDI port ${portIndex}`);
    } catch (err) {
      console.error('Failed to open MIDI port:', err);
    }
  }

  closeInput() {
    this.input.closePort();
  }
}
