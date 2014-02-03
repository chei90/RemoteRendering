
--position = {x = 10, z = 10, y = 0}
--w = IsWall(position);

math.randomseed(0)

g_dir = { x = 0, z = 1}

g_rotation = {x=0,y=0,z=0}

function PickDir()
    local dx = 0
    local dz = 0
    local radius = 0.5

	local r = math.random();
	
	if r < 0.5 then
		dx = -1
	else
		dx = 1
	end
	
	r = math.random();
	if r < 0.5 then
		dx = dx * radius
	else
		dx = 0
	end
	
	r = math.random();
	if dx == 0 then
		if r < 0.5 then
			dz = -1
		else
			dz = 1
		end
	end
	
	dz = dz * radius
	
    g_dir.x = dx
	g_dir.z = dz
end

function PrintVec(vec)
	Printf(vec.x);
	Printf(vec.z);
end

function ComputePosition(position)
	position.x = position.x + g_dir.x + ((g_dir.x > 0) and 1.5 or 0)
	position.z = position.z + g_dir.z + ((g_dir.z > 0) and 1.5 or 0)
	return position
end

function Adept(posNewPos)
	
	local dright = { x = 0, y = 0, z = 0 }
	local dleft = { x = 0, y = 0, z = 0 }
	local d = 0.5
	if g_dir.x > 0 then
		dleft.z = d
		dright.z = -d
	elseif g_dir.x < 0 then
		dleft.z = -d
		dright.z = d
	end
	
	if g_dir.z > 0 then
		dleft.x = -d
		dright.x = d
	elseif g_dir.z < 0 then
		dleft.x = d
		dright.x = -d
	end
	
	local left = { x = posNewPos.x + dleft.x, y = 0, z = posNewPos.z + dleft.z }
	local right = { x = posNewPos.x + dright.x, y = 0, z = posNewPos.z + dright.z }

	local leftw = IsWall(ComputePosition(left));
	local leftBack = { x = left.x + dleft.z, y = 0, z = left.z + dleft.x }

	local rightw = IsWall(ComputePosition(right));
	local rightBack = { x = right.x + dright.z, y = 0, z = right.z + dright.x }
	
	--[[if leftw == 0 then
		local iw = IsWall(ComputePosition(leftBack));
		if iw == 1 then
			g_dir.x = left.x - posNewPos.x
			g_dir.z = left.z - posNewPos.z
		end
	elseif rightw == 0 then
		local iw = IsWall(ComputePosition(rightBack));
		if iw == 1 then
			g_dir.x = right.x - posNewPos.x
			g_dir.z = right.z - posNewPos.z
		end
	end ]]
end
 
function OnUpdate(actorId, position, levelSize)
	local newPos = {x = 0, y = 0, z = 0}
	newPos.x = position.x + g_dir.x + ((g_dir.x > 0) and 1.5 or 0)
	newPos.z = position.z + g_dir.z + ((g_dir.z > 0) and 1.5 or 0)
	local w = IsWall(newPos);
	if w == 0 then
		--Adept(newPos);
		newPos.x = g_dir.x
		newPos.y = 0
		newPos.z = g_dir.z
		MoveActor(actorId, newPos, g_rotation, true);
	else
		PickDir();
	end
end 
