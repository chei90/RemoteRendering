
function EventListener(val)
	Printf("Hello");
	--position = {x = 0, y = 2, z = 0}
	--CreateActor("stairs.xml", position)
end

RegisterEventListener(EventType.ActorCreatedEvent, EventListener)
