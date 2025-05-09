obs = obslua
source_name = ""
start_time = 0
elapsed_time = 0
running = false

-- Description of the script
function script_description()
    return "A stopwatch with hotkeys for Start/Stop and Reset. Assign a text source to display the time."
end

-- Properties for the script
function script_properties()
    local props = obs.obs_properties_create()
    obs.obs_properties_add_text(props, "source_name", "Text Source Name", obs.OBS_TEXT_DEFAULT)
    return props
end

-- Update settings when they change
function script_update(settings)
    source_name = obs.obs_data_get_string(settings, "source_name")
end

-- Start or Stop the stopwatch
function toggle_timer(pressed)
    if not pressed then return end
    if running then
        running = false
        elapsed_time = elapsed_time + (os.time() - start_time)
    else
        running = true
        start_time = os.time()
    end
end

-- Reset the stopwatch
function reset_timer(pressed)
    if not pressed then return end
    running = false
    elapsed_time = 0
    update_text("00:00")
end

-- Update the text source with the current stopwatch time
function update_text(text)
    local source = obs.obs_get_source_by_name(source_name)
    if source then
        local settings = obs.obs_source_get_settings(source)
        obs.obs_data_set_string(settings, "text", text)
        obs.obs_source_update(source, settings)
        obs.obs_data_release(settings)
        obs.obs_source_release(source)
    end
end

-- Timer callback to refresh the displayed time
function tick()
    if running then
        local current_time = elapsed_time + (os.time() - start_time)
        local minutes = math.floor(current_time / 60)
        local seconds = current_time % 60
        update_text(string.format("%02d:%02d", minutes, seconds))
    end
end

-- Load the script and register hotkeys
function script_load(settings)
    obs.obs_hotkey_register_frontend("toggle_timer", "Start/Stop Stopwatch", toggle_timer)
    obs.obs_hotkey_register_frontend("reset_timer", "Reset Stopwatch", reset_timer)
    obs.timer_add(tick, 100) -- Update every 100ms
end

-- Cleanup on script unload
function script_unload()
    obs.timer_remove(tick)
end
