"""
logger_config.py
================
Centralized logging configuration for the bot.
"""

import logging
import logging.handlers
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file names with date rotation
TRADE_LOG = os.path.join(LOG_DIR, "trades.log")
BOT_LOG = os.path.join(LOG_DIR, "bot.log")
ERROR_LOG = os.path.join(LOG_DIR, "errors.log")

# Formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

simple_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Trade-specific formatter (CSV-like for easy analysis)
trade_formatter = logging.Formatter(
    '%(asctime)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class TelegramHandler(logging.Handler):
    """Custom logging handler that sends messages to Telegram"""
    
    def __init__(self, notifier, min_level=logging.WARNING):
        super().__init__()
        self.notifier = notifier
        self.min_level = min_level
        self.setLevel(min_level)
    
    def emit(self, record):
        """Send log record to Telegram"""
        try:
            # Format the message
            msg = self.format(record)
            
            # Only send trade-related messages (OPEN/CLOSE) or errors
            if ('OPEN|' in msg or 'CLOSE|' in msg or 
                record.levelno >= logging.ERROR or
                'TRADE' in msg.upper()):
                
                # Use async send if available, else sync
                if hasattr(self.notifier, 'send_message_async'):
                    import asyncio
                    asyncio.create_task(self.notifier.send_message_async(msg))
                else:
                    self.notifier.send_message_sync(msg)
        except Exception as e:
            # Don't let Telegram errors crash the logging
            print(f"Telegram logging error: {e}")


def setup_logging(verbose: bool = True, notifier=None):
    """Configure all loggers"""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (always show INFO and above)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console.setFormatter(simple_formatter)
    root_logger.addHandler(console)
    
    # File handler for all bot logs (rotating)
    bot_handler = logging.handlers.RotatingFileHandler(
        BOT_LOG,
        maxBytes=10_485_760,  # 10MB
        backupCount=5
    )
    bot_handler.setLevel(logging.DEBUG)
    bot_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(bot_handler)
    
    # File handler for errors only
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG,
        maxBytes=10_485_760,
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Special trade logger
    trade_logger = logging.getLogger('trades')
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False
    
    # Trade file handler
    trade_handler = logging.handlers.RotatingFileHandler(
        TRADE_LOG,
        maxBytes=10_485_760,
        backupCount=10
    )
    trade_handler.setLevel(logging.INFO)
    trade_handler.setFormatter(trade_formatter)
    trade_logger.addHandler(trade_handler)
    
    # Trade console handler (simple format)
    trade_console = logging.StreamHandler()
    trade_console.setLevel(logging.INFO)
    trade_console.setFormatter(logging.Formatter('%(message)s'))
    trade_logger.addHandler(trade_console)
    
    # Telegram handler for trade notifications (if notifier available)
    if notifier:
        try:
            telegram_handler = TelegramHandler(notifier, min_level=logging.INFO)
            telegram_handler.setFormatter(logging.Formatter('%(message)s'))
            
            # Add to trade logger for trade notifications
            trade_logger.addHandler(telegram_handler)
            
            # Also add to root logger for errors
            root_logger.addHandler(telegram_handler)
            
            logger = logging.getLogger(__name__)
            logger.info("✅ Telegram logging enabled")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Could not setup Telegram logging: {e}")
    
    return root_logger


def log_trade(action: str, **kwargs):
    """
    Log a trade in structured format for easy analysis.
    
    Example:
        log_trade('open', symbol='BTC/USDT', side='long', entry=50000, units=0.1)
        log_trade('close', symbol='BTC/USDT', pnl=500, pnl_pct=10, reason='take_profit')
    """
    trade_logger = logging.getLogger('trades')
    
    if action == 'open':
        msg = (f"OPEN|{kwargs.get('symbol')}|{kwargs.get('side')}|"
               f"{kwargs.get('entry')}|{kwargs.get('units')}|"
               f"{kwargs.get('stop_loss')}|{kwargs.get('take_profit')}")
    elif action == 'close':
        msg = (f"CLOSE|{kwargs.get('symbol')}|{kwargs.get('side')}|"
               f"{kwargs.get('entry')}|{kwargs.get('exit')}|"
               f"{kwargs.get('pnl')}|{kwargs.get('pnl_pct')}|"
               f"{kwargs.get('reason')}")
    else:
        msg = f"{action.upper()}|{kwargs}"
    
    trade_logger.info(msg)


# Global logger instance (will be configured later)
logger = None


def init_logging(verbose: bool = True, notifier=None):
    """Initialize logging with optional notifier"""
    global logger
    setup_logging(verbose=verbose, notifier=notifier)
    logger = logging.getLogger(__name__)
    return logger