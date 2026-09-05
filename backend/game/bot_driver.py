"""Shared bot action handling used by training and evaluation."""


def try_self_gang(game, bot, before_gang=None):
    """Take at most one accepted concealed/added kong; return whether taken.

    Call again before discarding after a replacement draw (which can also win).
    The order and kind match the web bot driver.
    """
    if game.phase != "discard_wait":
        return False
    seat = game.turn
    for tile in game._gang_options(seat):
        kind = "an" if game.players[seat].hand.count(tile) == 4 else "bu"
        if bot.decide_gang(tile, kind):
            if before_gang is not None:
                before_gang(seat, tile, kind)
            game.action_gang(seat, tile)
            return True
    return False


def finish_self_gangs(game, make_bot, trackers=()):
    """Resolve consecutive self-kongs, keeping optional public trackers in sync."""
    trackers = tuple(trackers)
    def notify(seat, tile, kind):
        for tracker in trackers:
            tracker.notify_self_gang(seat, tile, kind)
    while game.phase == "discard_wait":
        if not try_self_gang(game, make_bot(game, game.turn), notify):
            break
        if game.phase == "discard_wait":
            draw = game.last_drawn
            for tracker in trackers:
                tracker.notify_draw(draw['seat'], draw['tile'] if tracker.hero_seat == draw['seat'] else None,
                                    game.wall_remaining())
