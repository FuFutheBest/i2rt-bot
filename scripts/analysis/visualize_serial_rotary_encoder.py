# -*- coding: utf-8 -*-
"""
Serial Angle Reader with TUI Interface

Features:
- Open serial port with specified baudrate
- Send HEX data at 10Hz frequency
- Receive and parse angle data
- Display TUI interface with status information

Author: Han Xue
"""

import serial
import threading
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich import box
import keyboard
from loguru import logger


class AngleReader:
    def __init__(self, port: str = "COM4", baudrate: int = 1000000) -> None:
        """
        Initialize angle reader
        
        Args:
            port (str): Serial port name, default COM4
            baudrate (int): Baudrate, default 1M
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_running = False
        self.console = Console()
        
        # Communication data
        self.send_data = bytes.fromhex("01 03 00 41 00 01 d4 1e")
        self.last_received = b""
        self.last_angle = 0.0
        self.send_count = 0
        self.receive_count = 0
        self.error_count = 0
        
        # Status information
        self.status: Dict[str, Any] = {
            'port': port,
            'baudrate': baudrate,
            'connected': False,
            'last_send': '',
            'last_receive': '',
            'angle': 0.0,
            'angle_raw': 0,
            'send_count': 0,
            'receive_count': 0,
            'error_count': 0
        }

    
    def connect_serial(self) -> bool:
        """Connect to serial port"""
        self.serial_conn = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        self.status['connected'] = True
        return True
    
    def disconnect_serial(self) -> None:
        """Disconnect serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.status['connected'] = False
    
    def calculate_angle(self, data_bytes: bytes) -> float:
        """
        Calculate angle value from received data
        
        Args:
            data_bytes (bytes): Received data from serial port
            
        Returns:
            float: Calculated angle value in degrees
        """
        try:
            if len(data_bytes) >= 7:
                # Extract angle data (4th and 5th bytes)
                angle_high = data_bytes[3]
                angle_low = data_bytes[4]
                angle_raw = (angle_high << 8) | angle_low
                mask = 0b111111111111
                data_bit = 0b100000000000
                angle_raw = angle_raw & mask

                if (angle_raw & data_bit):
                    angle_raw = -((~angle_raw & mask) + 1)

                self.status['angle_raw'] = angle_raw
                
                # Calculate angle: 360 * raw_value / 4096
                angle = 360.0 * angle_raw / 4096.0
                return angle
        except Exception as e:
            self.error_count += 1
            return 0.0
        
        return 0.0
    
    def send_command(self) -> bool:
        """Send command to serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(self.send_data)
                self.send_count += 1
                self.status['send_count'] = self.send_count
                self.status['last_send'] = ' '.join([f"{b:02X}" for b in self.send_data])
                return True
            except Exception as e:
                self.error_count += 1
                return False
        return False
    
    def read_response(self) -> Optional[bytes]:
        """Read response from serial port"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # Try to read data
                data = self.serial_conn.read(10)  # Read up to 10 bytes
                if data:
                    self.last_received = data
                    self.receive_count += 1
                    self.status['receive_count'] = self.receive_count
                    self.status['last_receive'] = ' '.join([f"{b:02X}" for b in data])
                    
                    # Calculate angle
                    angle = self.calculate_angle(data)
                    self.last_angle = angle
                    self.status['angle'] = angle

                    return data
            except Exception as e:
                self.error_count += 1
                return None
        return None
    
    def communication_loop(self) -> None:
        """Communication loop - runs in separate thread"""
        while self.is_running:
            # Send command
            if self.send_command():
                self.read_response()
            
            # Update status
            self.status['error_count'] = self.error_count
    
    def create_layout(self) -> Layout:
        """Create TUI layout"""
        layout = Layout()
        
        # Main layout with three parts: title, angle display, status info
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=2),
            Layout(name="status", size=8)
        )
        
        # Status area split into two columns
        layout["status"].split_row(
            Layout(name="left_status"),
            Layout(name="right_status")
        )
        
        return layout
    
    def update_display(self, layout: Layout) -> None:
        """Update display content"""
        # Header
        layout["header"].update(
            Panel(
                Text("Serial Angle Reader - Press 'q' / Ctrl+C to exit", style="bold white", justify="center"),
                style="blue",
                box=box.ROUNDED
            )
        )
        
        # Main angle display
        angle_text = Text(f"{self.status['angle']:.3f}°", style="bold green")
        angle_text.stylize("bold", 0, len(angle_text))
        
        layout["main"].update(
            Panel(
                Text(f"{self.status['angle']:.3f}°", style="bold green", justify="center"),
                title="Current Angle",
                style="green",
                box=box.DOUBLE
            )
        )
        
        # Left status information
        left_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        left_table.add_column("Item", style="cyan")
        left_table.add_column("Value", style="white")
        
        connection_status = "✓ Connected" if self.status['connected'] else "✗ Disconnected"
        connection_style = "green" if self.status['connected'] else "red"
        
        left_table.add_row("Port:", f"{self.status['port']}")
        left_table.add_row("Baudrate:", f"{self.status['baudrate']}")
        left_table.add_row("Status:", Text(connection_status, style=connection_style))
        left_table.add_row("Send Count:", f"{self.status['send_count']}")
        left_table.add_row("Receive Count:", f"{self.status['receive_count']}")
        left_table.add_row("Error Count:", f"{self.status['error_count']}")
        
        layout["left_status"].update(
            Panel(
                left_table,
                title="Connection Status",
                style="blue"
            )
        )
        
        # Right communication data
        right_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        right_table.add_column("Item", style="cyan")
        right_table.add_column("Data", style="yellow")
        
        right_table.add_row("Send:", f"{self.status['last_send']}")
        right_table.add_row("Receive:", f"{self.status['last_receive']}")
        right_table.add_row("Raw Value:", f"{self.status['angle_raw']}")
        
        layout["right_status"].update(
            Panel(
                right_table,
                title="Communication Data",
                style="blue"
            )
        )
    
    def run(self) -> None:
        """Run main program"""
        logger.info("Starting serial angle reader...")

        # Connect to serial port
        if not self.connect_serial():
            logger.error("Unable to connect to serial port, exiting program")
            return
        
        logger.info(f"Serial port {self.port} connected successfully, baudrate: {self.baudrate}")
        
        # Create layout
        layout = self.create_layout()
        
        # Start communication thread
        self.is_running = True
        comm_thread = threading.Thread(target=self.communication_loop, daemon=True)
        comm_thread.start()
        
        try:
            # Use Rich Live display
            with Live(layout, refresh_per_second=100, screen=True) as live:
                while self.is_running:
                    self.update_display(layout)
                    live.update(layout)
                    
                    # Check for exit key
                    try:
                        if keyboard.is_pressed('q'):
                            break
                    except ImportError:
                        continue
        
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup resources
            self.is_running = False
            self.disconnect_serial()
            logger.info("Program exited")


def main() -> None:
    """Main function"""
    # Serial port parameters can be modified here
    # PORT = "COM4"        # Serial port on Windows
    PORT = "/dev/ttyUSB0"  # Serial port on Linux
    # If you cannot find this port, try `sudo apt-get remove brltty` to remove the brltty package
    BAUDRATE = 57600     # Baudrate (the same as bluetooth module)
    
    reader = AngleReader(port=PORT, baudrate=BAUDRATE)
    reader.run()


if __name__ == "__main__":
    main()